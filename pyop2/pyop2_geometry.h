///--- Computation of Jacobian matrices ---

/// Compute Jacobian J for interval embedded in R^1
#define compute_jacobian_interval_1d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0];

/// Compute Jacobian J for interval embedded in R^2
#define compute_jacobian_interval_2d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[3][0] - vertex_coordinates[2][0];

/// Compute Jacobian J for interval embedded in R^3
#define compute_jacobian_interval_3d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[3][0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[4][0] - vertex_coordinates[1][0]; \
  J[2] = vertex_coordinates[5][0] - vertex_coordinates[2][0];

/// Compute Jacobian J for triangle embedded in R^2
#define compute_jacobian_triangle_2d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; \
  J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; \
  J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0];

/// Compute Jacobian J for triangle embedded in R^3
#define compute_jacobian_triangle_3d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; \
  J[2] = vertex_coordinates[4][0] - vertex_coordinates[3][0]; \
  J[3] = vertex_coordinates[5][0] - vertex_coordinates[3][0]; \
  J[4] = vertex_coordinates[7][0] - vertex_coordinates[6][0]; \
  J[5] = vertex_coordinates[8][0] - vertex_coordinates[6][0];

/// Compute Jacobian J for tetrahedron embedded in R^3
#define compute_jacobian_tetrahedron_3d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[1] [0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[2] [0] - vertex_coordinates[0][0]; \
  J[2] = vertex_coordinates[3] [0] - vertex_coordinates[0][0]; \
  J[3] = vertex_coordinates[5] [0] - vertex_coordinates[4][0]; \
  J[4] = vertex_coordinates[6] [0] - vertex_coordinates[4][0]; \
  J[5] = vertex_coordinates[7] [0] - vertex_coordinates[4][0]; \
  J[6] = vertex_coordinates[9] [0] - vertex_coordinates[8][0]; \
  J[7] = vertex_coordinates[10][0] - vertex_coordinates[8][0]; \
  J[8] = vertex_coordinates[11][0] - vertex_coordinates[8][0];

/// Compute Jacobian J for tensor product prism embedded in R^3
#define compute_jacobian_prism_3d(J, vertex_coordinates) \
  J[0] = vertex_coordinates[2][0] - vertex_coordinates[0][0]; \
  J[1] = vertex_coordinates[4][0] - vertex_coordinates[0][0]; \
  J[2] = vertex_coordinates[1][0] - vertex_coordinates[0][0]; \
  J[3] = vertex_coordinates[2][1] - vertex_coordinates[0][1]; \
  J[4] = vertex_coordinates[4][1] - vertex_coordinates[0][1]; \
  J[5] = vertex_coordinates[1][1] - vertex_coordinates[0][1]; \
  J[6] = vertex_coordinates[2][2] - vertex_coordinates[0][2]; \
  J[7] = vertex_coordinates[4][2] - vertex_coordinates[0][2]; \
  J[8] = vertex_coordinates[1][2] - vertex_coordinates[0][2];

//--- Computation of Jacobian inverses ---

/// Compute Jacobian inverse K for interval embedded in R^1
#define compute_jacobian_inverse_interval_1d(K, det, J) \
  det = J[0]; \
  K[0] = 1.0 / det;

/// Compute Jacobian (pseudo)inverse K for interval embedded in R^2
#define compute_jacobian_inverse_interval_2d(K, det, J) \
  const double det2 = J[0]*J[0] + J[1]*J[1]; \
  det = sqrt(det2); \
  K[0] = J[0] / det2; \
  K[1] = J[1] / det2; \

/// Compute Jacobian (pseudo)inverse K for interval embedded in R^3
#define compute_jacobian_inverse_interval_3d(K, det, J) \
  const double det2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2]; \
  det = sqrt(det2); \
  K[0] = J[0] / det2; \
  K[1] = J[1] / det2; \
  K[2] = J[2] / det2;

/// Compute Jacobian inverse K for triangle embedded in R^2
#define compute_jacobian_inverse_triangle_2d(K, det, J) \
  det = J[0]*J[3] - J[1]*J[2]; \
  K[0] =  J[3] / det; \
  K[1] = -J[1] / det; \
  K[2] = -J[2] / det; \
  K[3] =  J[0] / det;

/// Compute Jacobian (pseudo)inverse K for triangle embedded in R^3
#define compute_jacobian_inverse_triangle_3d(K, det, J) \
  const double d_0 = J[2]*J[5] - J[4]*J[3]; \
  const double d_1 = J[4]*J[1] - J[0]*J[5]; \
  const double d_2 = J[0]*J[3] - J[2]*J[1]; \
  const double c_0 = J[0]*J[0] + J[2]*J[2] + J[4]*J[4]; \
  const double c_1 = J[1]*J[1] + J[3]*J[3] + J[5]*J[5]; \
  const double c_2 = J[0]*J[1] + J[2]*J[3] + J[4]*J[5]; \
  const double den = c_0*c_1 - c_2*c_2; \
  const double det2 = d_0*d_0 + d_1*d_1 + d_2*d_2; \
  det = sqrt(det2); \
  K[0] = (J[0]*c_1 - J[1]*c_2) / den; \
  K[1] = (J[2]*c_1 - J[3]*c_2) / den; \
  K[2] = (J[4]*c_1 - J[5]*c_2) / den; \
  K[3] = (J[1]*c_0 - J[0]*c_2) / den; \
  K[4] = (J[3]*c_0 - J[2]*c_2) / den; \
  K[5] = (J[5]*c_0 - J[4]*c_2) / den;

/// Compute Jacobian inverse K for tetrahedron embedded in R^3
#define compute_jacobian_inverse_tetrahedron_3d(K, det, J) \
  const double d_00 = J[4]*J[8] - J[5]*J[7]; \
  const double d_01 = J[5]*J[6] - J[3]*J[8]; \
  const double d_02 = J[3]*J[7] - J[4]*J[6]; \
  const double d_10 = J[2]*J[7] - J[1]*J[8]; \
  const double d_11 = J[0]*J[8] - J[2]*J[6]; \
  const double d_12 = J[1]*J[6] - J[0]*J[7]; \
  const double d_20 = J[1]*J[5] - J[2]*J[4]; \
  const double d_21 = J[2]*J[3] - J[0]*J[5]; \
  const double d_22 = J[0]*J[4] - J[1]*J[3]; \
  det = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; \
  K[0] = d_00 / det; \
  K[1] = d_10 / det; \
  K[2] = d_20 / det; \
  K[3] = d_01 / det; \
  K[4] = d_11 / det; \
  K[5] = d_21 / det; \
  K[6] = d_02 / det; \
  K[7] = d_12 / det; \
  K[8] = d_22 / det;
  
/// Compute Jacobian inverse K for tensor product prism embedded in R^3 - identical to tet
#define compute_jacobian_inverse_prism_3d(K, det, J) \
  const double d_00 = J[4]*J[8] - J[5]*J[7]; \
  const double d_01 = J[5]*J[6] - J[3]*J[8]; \
  const double d_02 = J[3]*J[7] - J[4]*J[6]; \
  const double d_10 = J[2]*J[7] - J[1]*J[8]; \
  const double d_11 = J[0]*J[8] - J[2]*J[6]; \
  const double d_12 = J[1]*J[6] - J[0]*J[7]; \
  const double d_20 = J[1]*J[5] - J[2]*J[4]; \
  const double d_21 = J[2]*J[3] - J[0]*J[5]; \
  const double d_22 = J[0]*J[4] - J[1]*J[3]; \
  det = J[0]*d_00 + J[3]*d_10 + J[6]*d_20; \
  K[0] = d_00 / det; \
  K[1] = d_10 / det; \
  K[2] = d_20 / det; \
  K[3] = d_01 / det; \
  K[4] = d_11 / det; \
  K[5] = d_21 / det; \
  K[6] = d_02 / det; \
  K[7] = d_12 / det; \
  K[8] = d_22 / det;
