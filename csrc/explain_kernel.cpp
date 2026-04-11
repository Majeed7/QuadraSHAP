#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

// Maximum supported quadrature order for stack-allocated scratch buffers in
// the quadrature-tree kernel. m_q = ceil(D/2) where D is the longest distinct
// feature path; in practice m_q never approaches this bound.
static constexpr int QT_MAX_MQ = 64;

struct PreparedTreeData {
    py::array_t<int32_t, py::array::c_style> feature_ids; // (n_leaves, D)
    py::array_t<double, py::array::c_style> lower;        // (n_leaves, D)
    py::array_t<double, py::array::c_style> upper;        // (n_leaves, D)
    py::array_t<double, py::array::c_style> invw;         // (n_leaves, D)
    py::array_t<double, py::array::c_style> alpha;        // (n_leaves, n_outputs)
    py::array_t<double, py::array::c_style> quad_x;       // (m_q,)
    py::array_t<double, py::array::c_style> quad_log_w;   // (m_q,)
    int n_leaves;
    int max_d;
};

void explain_trees(
    py::array_t<double, py::array::c_style> X,     // (n_samples, n_features)
    py::array_t<double, py::array::c_style> out,   // (n_samples, n_features, n_outputs)
    std::vector<PreparedTreeData>& trees,
    int n_trees)
{
    // Access raw pointers for maximum performance — avoid pybind11 accessor overhead
    const double* X_ptr = X.data();
    double* out_ptr = out.mutable_data();

    const ssize_t n_samples = X.shape(0);
    const ssize_t n_features = X.shape(1);
    const ssize_t n_outputs = out.shape(2);
    const ssize_t out_stride_s = n_features * n_outputs;
    const ssize_t out_stride_f = n_outputs;

    for (int t = 0; t < n_trees; t++) {
        const auto& tree = trees[t];
        const int D = tree.max_d;
        if (D == 0) continue;

        const int32_t* feat_ptr = tree.feature_ids.data();
        const double* lower_ptr = tree.lower.data();
        const double* upper_ptr = tree.upper.data();
        const double* invw_ptr = tree.invw.data();
        const double* alpha_ptr = tree.alpha.data();
        const double* qx_ptr = tree.quad_x.data();
        const double* qlw_ptr = tree.quad_log_w.data();

        const int n_leaves = tree.n_leaves;
        const int m_q = static_cast<int>(tree.quad_x.shape(0));

        // Pre-compute actual_d for each leaf (avoid repeated scanning)
        std::vector<int> leaf_actual_d(n_leaves);
        for (int leaf = 0; leaf < n_leaves; leaf++) {
            const int32_t* frow = feat_ptr + leaf * D;
            int ad = 0;
            for (int j = 0; j < D; j++) {
                if (frow[j] < 0) break;
                ad++;
            }
            leaf_actual_d[leaf] = ad;
        }

        // Reusable scratch buffers (allocated once per tree, sized for max D)
        std::vector<double> K(D);
        std::vector<int> feat(D);
        std::vector<double> Phi(D);
        std::vector<double> log_B(D);

        for (ssize_t s = 0; s < n_samples; s++) {
            const double* x_row = X_ptr + s * n_features;
            double* out_s = out_ptr + s * out_stride_s;

            for (int leaf = 0; leaf < n_leaves; leaf++) {
                const int actual_d = leaf_actual_d[leaf];
                if (actual_d == 0) continue;

                const int leaf_off = leaf * D;
                const int32_t* f_row = feat_ptr + leaf_off;
                const double* lo_row = lower_ptr + leaf_off;
                const double* hi_row = upper_ptr + leaf_off;
                const double* iw_row = invw_ptr + leaf_off;
                const double* al_row = alpha_ptr + leaf * n_outputs;

                // Compute K for each feature
                for (int j = 0; j < actual_d; j++) {
                    int fid = f_row[j];
                    feat[j] = fid;
                    double x_val = x_row[fid];
                    double q = (x_val > lo_row[j] && x_val <= hi_row[j]) ? iw_row[j] : 0.0;
                    K[j] = q - 1.0;
                }

                // Zero Phi
                std::memset(Phi.data(), 0, actual_d * sizeof(double));

                // Quadrature loop
                for (int qi = 0; qi < m_q; qi++) {
                    double t_val = qx_ptr[qi];
                    double log_w = qlw_ptr[qi];

                    double total_log = 0.0;
                    for (int j = 0; j < actual_d; j++) {
                        double lb = std::log1p(t_val * K[j]);
                        log_B[j] = lb;
                        total_log += lb;
                    }

                    double base = log_w + total_log;
                    for (int j = 0; j < actual_d; j++) {
                        Phi[j] += std::exp(base - log_B[j]);
                    }
                }

                // Accumulate Phi * K * alpha into output
                if (n_outputs == 1) {
                    // Fast path for single-output models
                    double a0 = al_row[0];
                    for (int j = 0; j < actual_d; j++) {
                        out_s[feat[j] * out_stride_f] += Phi[j] * K[j] * a0;
                    }
                } else {
                    for (int j = 0; j < actual_d; j++) {
                        double pk = Phi[j] * K[j];
                        double* dst = out_s + feat[j] * out_stride_f;
                        for (ssize_t o = 0; o < n_outputs; o++) {
                            dst[o] += pk * al_row[o];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Quadrature-tree kernel: O(m_q * L) two-pass algorithm from paper.tex.
// ---------------------------------------------------------------------------

struct PreparedTreeQT {
    py::array_t<int32_t, py::array::c_style> children_left;   // (n_nodes,)
    py::array_t<int32_t, py::array::c_style> children_right;  // (n_nodes,)
    py::array_t<int32_t, py::array::c_style> feature;         // (n_nodes,)
    py::array_t<double, py::array::c_style> threshold;        // (n_nodes,)
    py::array_t<double, py::array::c_style> edge_weight;      // (n_nodes,) NaN at root
    py::array_t<double, py::array::c_style> values;           // (n_nodes, n_outputs)
    py::array_t<int32_t, py::array::c_style> postorder;       // (n_nodes,)
    py::array_t<double, py::array::c_style> quad_x;           // (m_q,)
    py::array_t<double, py::array::c_style> quad_w;           // (m_q,)
    int n_features;
    int n_outputs;
    double tree_weight;
};

namespace {

// Per-tree, per-sample worker. State is held as raw pointers + scratch buffers
// reused across samples. The first pass is a recursive DFS that fills G_node
// and Delta_F; the second pass walks the post-order and accumulates Shapley
// values into out_s.
struct QuadratureTreeWorker {
    // Tree topology / params (raw pointers into the PreparedTreeQT arrays).
    const int32_t* cl;
    const int32_t* cr;
    const int32_t* feat;
    const double* thr;
    const double* ew;
    const double* tree_values;
    const int32_t* postorder;
    const double* qx;
    const double* qw;
    int n_nodes;
    int n_features;
    int n_outputs;
    int m_q;
    double tree_weight;

    // Per-sample inputs / outputs.
    const double* x_row;
    double* out_s;
    ssize_t out_stride_f; // = n_outputs of the *output* array (may differ from
                          // tree.n_outputs in degenerate cases, but for our
                          // use they match).

    // Scratch state, sized for the current tree.
    std::vector<double> G_node;     // (n_nodes * m_q)
    std::vector<double> Delta_F;    // (n_nodes * m_q)
    std::vector<double> H;          // (n_nodes * m_q * n_outputs)
    std::vector<double> current_q;  // (n_features,)
    std::vector<double> current_F;  // (n_features * m_q)
    std::vector<double> G_cur;      // (m_q,)

    void resize_for_tree() {
        G_node.assign(static_cast<size_t>(n_nodes) * m_q, 0.0);
        Delta_F.assign(static_cast<size_t>(n_nodes) * m_q, 0.0);
        H.assign(static_cast<size_t>(n_nodes) * m_q * n_outputs, 0.0);
        current_q.assign(n_features, 1.0);
        current_F.assign(static_cast<size_t>(n_features) * m_q, 0.0);
        G_cur.assign(m_q, 1.0);
    }

    // Re-initialize the per-sample mutable buffers without reallocating.
    void reset_for_sample() {
        std::fill(current_q.begin(), current_q.end(), 1.0);
        std::fill(current_F.begin(), current_F.end(), 0.0);
        std::fill(G_cur.begin(), G_cur.end(), 1.0);
        // G_node, Delta_F, H are fully overwritten in each pass; no reset
        // needed for correctness, but the second pass relies on Delta_F being
        // 0 at the root slot (postorder edge accumulation skips the root, so
        // the value is never read; we leave it untouched).
    }

    // First pass: descend the tree, populating G_node[u, :] and
    // Delta_F[child, :] for every non-root node.
    void dfs1(int u) {
        // Snapshot G_cur into G_node[u, :].
        std::memcpy(G_node.data() + static_cast<size_t>(u) * m_q,
                    G_cur.data(), m_q * sizeof(double));

        const int l = cl[u];
        if (l == -1) {
            return;
        }
        const int r = cr[u];
        const int f = feat[u];
        const double thr_val = thr[u];

        const double q_old = current_q[f];

        double F_old[QT_MAX_MQ];
        double a_old[QT_MAX_MQ];
        double G_saved[QT_MAX_MQ];

        double* curF_f = current_F.data() + static_cast<size_t>(f) * m_q;
        std::memcpy(F_old, curF_f, m_q * sizeof(double));
        std::memcpy(G_saved, G_cur.data(), m_q * sizeof(double));
        for (int ri = 0; ri < m_q; ri++) {
            a_old[ri] = (1.0 - qx[ri]) + qx[ri] * q_old;
        }

        const double x_f = x_row[f];
        const double sat_left = (x_f <= thr_val) ? 1.0 : 0.0;
        const int children[2] = {l, r};
        const double sats[2] = {sat_left, 1.0 - sat_left};

        for (int c = 0; c < 2; c++) {
            const int child = children[c];
            const double sat = sats[c];
            const double w_e = ew[child];
            const double inv_w = 1.0 / w_e;

            const double q_new = q_old * inv_w * sat;
            current_q[f] = q_new;

            double* dF_c = Delta_F.data() + static_cast<size_t>(child) * m_q;
            for (int ri = 0; ri < m_q; ri++) {
                const double a_new = (1.0 - qx[ri]) + qx[ri] * q_new;
                const double F_new = (q_new - 1.0) / a_new;
                dF_c[ri] = F_new - F_old[ri];
                curF_f[ri] = F_new;
                G_cur[ri] = G_saved[ri] * w_e * (a_new / a_old[ri]);
            }

            dfs1(child);
        }

        // Restore parent state.
        current_q[f] = q_old;
        std::memcpy(curF_f, F_old, m_q * sizeof(double));
        std::memcpy(G_cur.data(), G_saved, m_q * sizeof(double));
    }

    // Second pass: bottom-up via postorder. Computes H[u] and, at every
    // internal node, accumulates the per-feature Shapley contributions for
    // both outgoing edges.
    void second_pass() {
        for (int idx = 0; idx < n_nodes; idx++) {
            const int u = postorder[idx];
            const int l = cl[u];

            double* H_u = H.data() + (static_cast<size_t>(u) * m_q) * n_outputs;

            if (l == -1) {
                // Leaf: H[u, r, k] = G_node[u, r] * V_u[k]
                const double* gn = G_node.data() + static_cast<size_t>(u) * m_q;
                const double* V_u = tree_values + static_cast<size_t>(u) * n_outputs;
                for (int ri = 0; ri < m_q; ri++) {
                    const double g = gn[ri];
                    double* H_row = H_u + ri * n_outputs;
                    for (int k = 0; k < n_outputs; k++) {
                        H_row[k] = g * V_u[k];
                    }
                }
                continue;
            }

            const int r_node = cr[u];
            // H[u] = H[l] + H[r]
            const double* H_l = H.data() + (static_cast<size_t>(l) * m_q) * n_outputs;
            const double* H_r = H.data() + (static_cast<size_t>(r_node) * m_q) * n_outputs;
            const size_t hu_len = static_cast<size_t>(m_q) * n_outputs;
            for (size_t i = 0; i < hu_len; i++) {
                H_u[i] = H_l[i] + H_r[i];
            }

            // Edge contributions for both children:
            //   phi[f] += sum_r tree_weight * qw[r] * Delta_F[child, r] * H[child, r, :]
            const int f = feat[u];
            double* phi_f = out_s + static_cast<ssize_t>(f) * out_stride_f;
            const int children[2] = {l, r_node};
            for (int c = 0; c < 2; c++) {
                const int child = children[c];
                const double* H_c = H.data() + (static_cast<size_t>(child) * m_q) * n_outputs;
                const double* dF_c = Delta_F.data() + static_cast<size_t>(child) * m_q;
                if (n_outputs == 1) {
                    double acc = 0.0;
                    for (int ri = 0; ri < m_q; ri++) {
                        acc += qw[ri] * dF_c[ri] * H_c[ri];
                    }
                    phi_f[0] += tree_weight * acc;
                } else {
                    for (int ri = 0; ri < m_q; ri++) {
                        const double w = tree_weight * qw[ri] * dF_c[ri];
                        const double* H_row = H_c + ri * n_outputs;
                        for (int k = 0; k < n_outputs; k++) {
                            phi_f[k] += w * H_row[k];
                        }
                    }
                }
            }
        }
    }
};

}  // namespace

void explain_trees_quadrature(
    py::array_t<double, py::array::c_style> X,    // (n_samples, n_features)
    py::array_t<double, py::array::c_style> out,  // (n_samples, n_features, n_outputs)
    std::vector<PreparedTreeQT>& trees,
    int n_trees)
{
    const double* X_ptr = X.data();
    double* out_ptr = out.mutable_data();

    const ssize_t n_samples = X.shape(0);
    const ssize_t n_features = X.shape(1);
    const ssize_t n_outputs = out.shape(2);
    const ssize_t out_stride_s = n_features * n_outputs;
    const ssize_t out_stride_f = n_outputs;

    QuadratureTreeWorker w;

    for (int t = 0; t < n_trees; t++) {
        const auto& tree = trees[t];
        const int n_nodes = static_cast<int>(tree.children_left.shape(0));
        if (n_nodes == 0) continue;
        const int m_q = static_cast<int>(tree.quad_x.shape(0));
        if (m_q > QT_MAX_MQ) {
            throw std::runtime_error(
                "explain_trees_quadrature: m_q exceeds QT_MAX_MQ");
        }
        if (tree.n_outputs != n_outputs) {
            throw std::runtime_error(
                "explain_trees_quadrature: per-tree n_outputs must match output array");
        }

        w.cl = tree.children_left.data();
        w.cr = tree.children_right.data();
        w.feat = tree.feature.data();
        w.thr = tree.threshold.data();
        w.ew = tree.edge_weight.data();
        w.tree_values = tree.values.data();
        w.postorder = tree.postorder.data();
        w.qx = tree.quad_x.data();
        w.qw = tree.quad_w.data();
        w.n_nodes = n_nodes;
        w.n_features = tree.n_features;
        w.n_outputs = tree.n_outputs;
        w.m_q = m_q;
        w.tree_weight = tree.tree_weight;
        w.resize_for_tree();

        for (ssize_t s = 0; s < n_samples; s++) {
            w.x_row = X_ptr + s * n_features;
            w.out_s = out_ptr + s * out_stride_s;
            w.out_stride_f = out_stride_f;

            w.reset_for_sample();
            w.dfs1(0);
            w.second_pass();
        }
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ acceleration for pgshapley TreeSHAP";

    py::class_<PreparedTreeData>(m, "PreparedTreeData")
        .def(py::init<>())
        .def_readwrite("feature_ids", &PreparedTreeData::feature_ids)
        .def_readwrite("lower", &PreparedTreeData::lower)
        .def_readwrite("upper", &PreparedTreeData::upper)
        .def_readwrite("invw", &PreparedTreeData::invw)
        .def_readwrite("alpha", &PreparedTreeData::alpha)
        .def_readwrite("quad_x", &PreparedTreeData::quad_x)
        .def_readwrite("quad_log_w", &PreparedTreeData::quad_log_w)
        .def_readwrite("n_leaves", &PreparedTreeData::n_leaves)
        .def_readwrite("max_d", &PreparedTreeData::max_d);

    m.def("explain_trees", &explain_trees,
          "Compute SHAP values for all trees (C++ hot loop)",
          py::arg("X"), py::arg("out"), py::arg("trees"), py::arg("n_trees"));

    py::class_<PreparedTreeQT>(m, "PreparedTreeQT")
        .def(py::init<>())
        .def_readwrite("children_left", &PreparedTreeQT::children_left)
        .def_readwrite("children_right", &PreparedTreeQT::children_right)
        .def_readwrite("feature", &PreparedTreeQT::feature)
        .def_readwrite("threshold", &PreparedTreeQT::threshold)
        .def_readwrite("edge_weight", &PreparedTreeQT::edge_weight)
        .def_readwrite("values", &PreparedTreeQT::values)
        .def_readwrite("postorder", &PreparedTreeQT::postorder)
        .def_readwrite("quad_x", &PreparedTreeQT::quad_x)
        .def_readwrite("quad_w", &PreparedTreeQT::quad_w)
        .def_readwrite("n_features", &PreparedTreeQT::n_features)
        .def_readwrite("n_outputs", &PreparedTreeQT::n_outputs)
        .def_readwrite("tree_weight", &PreparedTreeQT::tree_weight);

    m.def("explain_trees_quadrature", &explain_trees_quadrature,
          "Compute SHAP values via the O(m_q*L) two-pass quadrature kernel",
          py::arg("X"), py::arg("out"), py::arg("trees"), py::arg("n_trees"));
}
