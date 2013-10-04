/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    amcgsoftware@imperial.ac.uk

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/



#include <confdefs.h>

#include <cmath>
#include <iostream>
#include <cassert>
#include <getopt.h>
#include <string>
#include <map>
#include <vector>
#include "global_parameters.h"

using namespace std;

#ifndef PI
#define PI 3.1415926535897931
#endif

#ifndef RAD_TO_DEG_FACTOR
#define RAD_TO_DEG_FACTOR 57.2957795147199532
#endif

#ifndef DEG_TO_RAD_FACTOR
#define DEG_TO_RAD_FACTOR 0.0174532925194444
#endif

int stereographic2spherical(double x, double y, double& longitude, double& latitude);
void tests();

// fortran wrappers - we do *not* want to start messing with fortran chars to C++ strings, so wrap some of the functions
// up
extern "C" {
#define projections_spherical_cartesian_fc F77_FUNC(projections_spherical_cartesian, PROJECTIONS_SPHERICAL_CARTESIAN)
    int projections_spherical_cartesian_fc(int *nPoints, double *x, double *y, double *z);

#define projections_cartesian_spherical_fc F77_FUNC(projections_cartesian_spherical, PROJECTIONS_CARTESIAN_SPHERICAL)
    int projections_cartesian_spherical_fc(int *nPoints, double *x, double *y, double *z);

}

enum ExitCodes{
  SUCCESS = 0,
  BAD_ARGUMENTS = -1,
  UNKNOWN_CELL_TYPE = -2
};

double degrees(double radians){
  return radians*RAD_TO_DEG_FACTOR;
}

double radians(double degrees){
  return degrees*DEG_TO_RAD_FACTOR;
}

int spherical2cartesian(double longitude, double latitude, double &x, double &y, double &z){
  // latitude  = 90.0 - latitude;
  latitude  = radians(latitude);
  longitude = radians(longitude);

  x = get_surface_radius()*cos(longitude)*sin(PI*0.5-latitude);
  y = get_surface_radius()*sin(longitude)*sin(PI*0.5-latitude);
  z = get_surface_radius()*cos(PI*0.5-latitude);

  return 0;
}

int cartesian2spherical(double x, double y, double z, double &longitude, double &latitude) {

    double r = sqrt(x*x + y*y + z*z);

    longitude = atan2(y,x);
    latitude = acos(z/r);
    // convert to degrees
    latitude = degrees(latitude);
    latitude = 90.0-latitude;
    longitude = degrees(longitude);

    return 0;
}

int spherical2stereographic(double longitude, double latitude, double &x, double &y){
  // http://mathworld.wolfram.com/StereographicProjection.html
  longitude = radians(longitude);
  latitude  = radians(latitude);

  double longitude_0 = 0.0;
  double latitude_0  = radians(90.0);

  if(latitude_0==latitude){
    x = 0.0;
    y = 0.0;
    return -1;
  }

  double k = 2.0*get_surface_radius()/
    (1.0 +
     sin(latitude_0)*sin(latitude) +
     cos(latitude_0)*cos(latitude)*cos(longitude-longitude_0));

   x = k*cos(latitude)*sin(longitude-longitude_0);
   y = k*(cos(latitude_0)*sin(latitude) -
       sin(latitude_0)*cos(latitude)*cos(longitude-longitude_0));

  return 0;
}

int stereographic2cartesian(double xx, double yy, double &x, double &y, double &z){
  double longitude, latitude;
  stereographic2spherical(xx, yy, longitude, latitude);
  spherical2cartesian(longitude, latitude, x, y, z);
  return 0;
}

int stereographic2spherical(double x, double y, double& longitude, double& latitude){
  // http://mathworld.wolfram.com/StereographicProjection.html
  double longitude_0 = 0.0;
  double latitude_0  = radians(90.0);

  if((x==0.0)&&(y==0.0)){
    latitude  = degrees(latitude_0);
    longitude = degrees(longitude_0);
    return 0;
  }

  double rho = sqrt(x*x + y*y);
  double c = 2.0*atan2(rho, 2*get_surface_radius());

  latitude = asin(cos(c)*sin(latitude_0) + y*sin(c)*cos(latitude_0)/rho);
  longitude = longitude_0 + atan2(x*sin(c),
          rho*cos(latitude_0)*cos(c) -
          y*sin(latitude_0)*sin(c));

  latitude  = degrees(latitude);
  longitude = degrees(longitude);

  return 0;
}

int projections(int nPoints, double *x, double *y, double *z, string current_coord, string output_coord){


    for (int i=0; i<nPoints; i++) {
        double new_x = 0.0;
        double new_y = 0.0;
        double new_z = 0.0;

        if(current_coord == "stereo" && output_coord == "cart") {
            stereographic2cartesian(x[i], y[i], new_x, new_y, new_z);
            x[i] = new_x;
            y[i] = new_y;
            z[i] = new_z;
        } else if (current_coord == "spherical" && output_coord == "cart") {
            spherical2cartesian(x[i], y[i], new_x, new_y, new_z);
            x[i] = new_x;
            y[i] = new_y;
            z[i] = new_z;
        } else if (current_coord == "spherical" && output_coord == "stereo") {
            spherical2stereographic(x[i], y[i], new_x, new_y);
            x[i] = new_x;
            y[i] = new_y;
            z[i] = 0.0;
        } else if (current_coord == "stereo" && output_coord == "spherical") {
            stereographic2spherical(x[i],y[i],new_x,new_y);
            x[i] = new_x;
            y[i] = new_y;
            z[i] = 0.0;
        } else if (current_coord == "cart" && output_coord == "spherical") {
            cartesian2spherical(x[i], y[i], z[i], new_x, new_y);
            x[i] = new_x;
            y[i] = new_y;
            z[i] = 0.0;
        } else {
            cerr<<"ERROR: transformation not implemented: "<<current_coord<<" --> "<<output_coord<<"\n";
            return -1;
        }
    }

    return 0;
}

/*
 * Fortran wrappers
 */
int projections_spherical_cartesian_fc(int *nPoints, double *x, double *y, double *z) {
    projections(*nPoints,x,y,z,"spherical","cart");
    return 0;
}

int projections_cartesian_spherical_fc(int *nPoints, double *x, double *y, double *z) {
    projections(*nPoints,x,y,z,"cart","sperical");
    return 0;
}


#ifdef PROJECTIONS_UNIT_TEST
#include <vtk.h>

int main(int argc, char **argv){

    char *filename = argv[1];
    char *filename_out = argv[2];

    // first run the Noddy tests
    tests();

    // now run a test on a vtk file
    vtkXMLUnstructuredGridReader* reader = vtkXMLUnstructuredGridReader::New();

    reader->SetFileName(filename);
    reader->Update();

    vtkUnstructuredGrid* grid = vtkUnstructuredGrid::New();
    grid->DeepCopy(reader->GetOutput());
    grid->Update();

    reader->Delete();
    int NNodes = grid->GetNumberOfPoints();

    vector<double> x(NNodes,0.0), y(NNodes,0.0), z(NNodes,0.0);

    for(int i=0;i<NNodes;i++){
        double xyz[3];
        grid->GetPoints()->GetPoint(i, xyz);
        x[i] = xyz[0];
        y[i] = xyz[1];
        z[i] = xyz[2];
    }

    projections(NNodes, &x[0], &y[0], &z[0],"cart","spherical");

    for(int i=0;i<NNodes;i++){
        grid->GetPoints()->SetPoint(i, x[i], y[i], z[i]);
    }

    vtkXMLUnstructuredGridWriter* writer= vtkXMLUnstructuredGridWriter::New();
    writer->SetFileName(filename_out);
    writer->SetInput(grid);
    vtkZLibDataCompressor* compressor = vtkZLibDataCompressor::New();
    writer->SetCompressor(compressor);
    writer->Write();
    writer->Delete();
    compressor->Delete();

    return 0;
}

void tests(){
  double x0, y0, z0;
  double x1, y1, z1;
  double x2, y2, z2;

  cout<<"Testing stereographic2spherical/spherical2stereographic:"<<endl;
  x0=0; y0=0;
  stereographic2spherical(x0, y0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<" --> ";
  spherical2stereographic(x1, y1, x2, y2);
  cout<<x2<<", "<<y2<<endl;

  cout<<"Testing spherical2stereographic/stereographic2spherical:"<<endl;
  x0=45; y0=45;
  spherical2stereographic(x0, y0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<" --> ";
  stereographic2spherical(x1, y1, x2, y2);
  cout<<x2<<", "<<y2<<endl;

  cout<<"Testing spherical2stereographic/stereographic2spherical:"<<endl;
  x0=0; y0=0; z0=0;
  cartesian2spherical(x0, y0, z0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<endl;
  x0=0; y0=100; z0=0;
  cartesian2spherical(x0, y0, z0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<endl;
  x0=100; y0=0; z0=0;
  cartesian2spherical(x0, y0, z0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<endl;
  x0=100; y0=100; z0=0;
  cartesian2spherical(x0, y0, z0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<endl;

  x0=215.0; y0=54.0; z0=get_surface_radius();
  spherical2cartesian(x0,y0,x1,y1,z1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<", "<<z1<<endl;

  x0=x1; y0=y1; z0=z1;
  cartesian2spherical(x0, y0, z0, x1, y1);
  cout<<x0<<", "<<y0<<" --> "<<x1<<", "<<y1<<endl;


}

#endif
