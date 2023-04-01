#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

/* print_array prints an MxN array of type T.
 * 
 * \param [in] M:     number of rows in the array
 * \param [in] N:     number of columns in the array
 * \param [in] array: the array to be printed
 */
template <typename T> 
void print_array(int M, int N, const T *array) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << array[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


/* writeVTK writes the data in sol to .vtk format with timestamp time.
 *
 * \param [in] numElem: number of elements in the mesh
 * \param [in] numVert: number of nodes in the mesh
 * \param [in] EToV:    connectivity array of the mesh
 * \param [in] Vxy:     array of node coordinates
 * \param [in] sol:     array of data corresponding to either mesh cells or
 *                      mesh vertices
 * \param [in] time:    simulation time at which sol is associated with
 * \param [in] iter:    cycle number of filenaming purposes (output will be
 *                      named "results<iter>.vtk")
 */
template <typename T>
void writeVTK(int numElem, int numVert, const int *EToV, const T *Vxy, T *sol, double time, int iter) {
  const std::string name = "./results/results";
  ofstream myfile (name + std::to_string(iter) + ".vtk");

  if (myfile.is_open())
  {
    myfile << "# vtk DataFile Version 3.0\n";
    myfile << "vtk output\n";
    myfile << "ASCII\n";
    myfile << "DATASET UNSTRUCTURED_GRID\n";
    myfile << "FIELD " << "FieldData " << 2 << "\n";
    myfile << "TIME " << 1 << " " << 1 << " double\n";
    myfile <<  time << "\n";
    myfile << "CYCLE " << 1 << " " << 1 << " int\n";
    myfile << iter << "\n";
    myfile << "POINTS " << numVert << " double\n";

    // write vertex data (each row gives coordinates for the vertex)
    for(int i = 0; i < numVert; i++){
      myfile << Vxy[i*2] << " " << Vxy[i*2 + 1] << " " << 0 << "\n";
    }
    myfile << "\n";

    // write connectivity data (each row gives vertices of the 
    // element in counterclockwise ordering)
    myfile << "CELLS " << numElem << " " << 5*numElem << "\n";
    for (int i = 0; i < numElem; ++i)
    {
      myfile << 4 << " " << EToV[i*6] << " " << EToV[i*6 + 1] << " " << EToV[i*6 + 2] << " " << EToV[i*6 + 3] << "\n";
    }
    myfile << "\n";

    // vtk CELL_TYPE 9 corresponds to quadrilateral cells
    myfile << "CELL_TYPES " << numElem << "\n";
    for (int i = 0; i < numElem; ++i)
    {
      myfile << 9 << "\n";
    }
    myfile << "\n";

    // the LOOKUP_TABLE consists of one piece of data on each cell
    myfile << "CELL_DATA " << numElem << "\n";
    myfile << "SCALARS " << "density " << "double\n";
    myfile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < numElem; ++i)
    {
      myfile << sol[i] << "\n";
    }


    // myfile << "POINT_DATA " << numVert << "\n";
    // myfile << "SCALARS " << "solution " << "double\n";
    // myfile << "LOOKUP_TABLE default\n";
    // for (int i = 0; i < numVert; ++i)
    // {
    //   myfile << sol[i] << "\n";
    // }

    myfile.close();
  }
  else cout << "Unable to open file";
}

#endif