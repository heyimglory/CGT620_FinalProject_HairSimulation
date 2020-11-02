#ifndef __helper_h__
#define __helper_h__

#include "helper.h"
#include "math/vect3d.h"

/*********************************
Some OpenGL-related functions DO NOT TOUCH
**********************************/
//displays the text message in the GL window
void GLMessage(char *message);

//called when a window is reshaped
void Reshape(int w, int h);

//Some simple rendering routines using old fixed-pipeline OpenGL
//draws line from a to b with color 
void DrawLine(Vect3d a, Vect3d b, Vect3d color);

//draws point at a with color 
void DrawPoint(Vect3d a, Vect3d color);

//OpenGL freeglut stuff
void Idle(void);

//OpenGL freeglut stuff
void Display(void);


//display coordinate system
void CoordSyst();


#endif