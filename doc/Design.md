Design principles
====================

1. All arrays are Kokkos::View
2. All changes to a mesh (refinement, coarsening, reconnection, etc.) happen on the host only.
3. Main views are public, host views private
3. MeshSeeds know geometry and faces


To do
======================
1. Do faces need to know center indices, or are they 1-1 with face indices?

Status
======================
1. Verify/test faces

