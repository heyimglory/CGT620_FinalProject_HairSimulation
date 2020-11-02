#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <vector>			//Standard template library class
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//in house created libraries
#include "vect3d.h"
#include "helper.h"

#include "imgui_impl_glut.h"


#pragma comment(lib, "freeglut.lib")

GLint wWindow = 1200;
GLint hWindow = 800;

GLfloat light_pos[3] = { 0.0f, 20.0f, 0.0f };
const float box_size = 4.0f;
GLfloat box_color_a[4] = { 0.1f, 0.2f, 0.2f, 1.0f };
GLfloat box_color_d[4] = { 0.3f, 0.5f, 0.5f, 1.0f };

// camera ctrl
float cam_h = 0.0f;
float cam_v = 0.0f;
float cam_d = 2.0f * box_size;

// param
const int MAX_DENSITY = 10;
const int MAX_SEG_NUM = 50;
const int MAX_TESS = (MAX_DENSITY + 1) * (MAX_DENSITY + 2) / 2;
int hair_density = 1;
float hair_seg_length = 0.008f;
int hair_seg_num = 10;
float hair_stiff = 6.0f;
float gravity = 0.8f;
float damping = 0.7f;

std::vector<glm::vec3> sphere_vertices;
std::vector<glm::vec2> sphere_tex_coord;
std::vector<glm::vec3> sphere_normals;
int sphere_tri_num = 0;
int tess_num;

float old_time, cur_time, d_time;

// CUDA stuff
float *d_pos, *d_vel, *d_col;

// hair data
GLfloat pos[960][MAX_TESS][MAX_SEG_NUM][3];
GLfloat vel[960][MAX_TESS][MAX_SEG_NUM][3];
GLfloat col[960][MAX_TESS][MAX_SEG_NUM][3];

void compute_hair();

void Cleanup(bool noError)
{
	cudaError_t error;
	// Free device memory
	if (d_pos) error = cudaFree(d_pos);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
	if (d_vel) error = cudaFree(d_vel);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
	if (d_col) error = cudaFree(d_col);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
}

void load_obj(const char* filename, std::vector<glm::vec3> &vertices, std::vector<glm::vec2> &tex_coord, std::vector<glm::vec3> &normals)
{
	std::vector< unsigned int > vertexIndices, uvIndices, normalIndices;
	std::vector< glm::vec3 > temp_vertices;
	std::vector< glm::vec2 > temp_uvs;
	std::vector< glm::vec3 > temp_normals;

	FILE * file = fopen(filename, "r");
	if (file == NULL) {
		printf("Impossible to open the file !\n");
		return;
	}

	while (1)
	{
		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.
		if (strcmp(lineHeader, "v") == 0)
		{
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		}
		else if (strcmp(lineHeader, "vt") == 0) {
			glm::vec2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y);
			temp_uvs.push_back(uv);
		}
		else if (strcmp(lineHeader, "vn") == 0) {
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		}
		else if (strcmp(lineHeader, "f") == 0)
		{
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser : ( Try exporting with other options\n");
				return;
			}
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices.push_back(uvIndex[0]);
			uvIndices.push_back(uvIndex[1]);
			uvIndices.push_back(uvIndex[2]);
			normalIndices.push_back(normalIndex[0]);
			normalIndices.push_back(normalIndex[1]);
			normalIndices.push_back(normalIndex[2]);
		}
	}

	// For each vertex of each triangle
	for (unsigned int i = 0; i < vertexIndices.size(); i++)
	{
		unsigned int vertexIndex = vertexIndices[i];
		glm::vec3 vertex = temp_vertices[vertexIndex - 1];
		vertices.push_back(vertex);
	}
	// For each texcoord of each triangle
	for (unsigned int i = 0; i < uvIndices.size(); i++)
	{
		unsigned int uvIndex = uvIndices[i];
		glm::vec2 uvs = temp_uvs[uvIndex - 1];
		tex_coord.push_back(uvs);
	}
	// For each vertex of each triangle
	for (unsigned int i = 0; i < normalIndices.size(); i++)
	{
		unsigned int normalIndex = normalIndices[i];
		glm::vec3 normal = temp_normals[normalIndex - 1];
		normals.push_back(normal);
	}
	sphere_tri_num = vertexIndices.size() / 3;
}

void draw_box()
{
	glEnable(GL_LIGHTING);
	glMaterialfv(GL_FRONT, GL_AMBIENT, box_color_a);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, box_color_d);
	glMaterialf(GL_FRONT, GL_SPECULAR, 0.5);
	glDepthMask(GL_FALSE);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	// bottom
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3f(0.0, 1.0, 0.0);
	glVertex3f(box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(box_size, -1.0 * box_size, box_size);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, box_size);
	glEnd();
	// right
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3f(-1.0, 0.0, 0.0);
	glVertex3f(box_size, -1.0 * box_size, box_size);
	glVertex3f(box_size, box_size, box_size);
	glVertex3f(box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(box_size, box_size, -1.0 * box_size);
	glEnd();
	// back
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3f(0.0, 0.0, 1.0);
	glVertex3f(box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(box_size, box_size, -1.0 * box_size);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(-1.0 * box_size, box_size, -1.0 * box_size);
	glEnd();
	// left
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3f(1.0, 0.0, 0.0);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, -1.0 * box_size);
	glVertex3f(-1.0 * box_size, box_size, -1.0 * box_size);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, box_size);
	glVertex3f(-1.0 * box_size, box_size, box_size);
	glEnd();
	// front
	glBegin(GL_TRIANGLE_STRIP);
	glNormal3f(0.0, 0.0, -1.0);
	glVertex3f(-1.0 * box_size, -1.0 * box_size, box_size);
	glVertex3f(-1.0 * box_size, box_size, box_size);
	glVertex3f(box_size, -1.0 * box_size, box_size);
	glVertex3f(box_size, box_size, box_size);
	glEnd();
	glDisable(GL_CULL_FACE);
	glDepthMask(GL_TRUE);
	glDisable(GL_LIGHTING);
}

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();

	ImGui::SliderInt("Hair Density", &hair_density, 1, MAX_DENSITY);
	ImGui::SliderFloat("Hair Segment Length", &hair_seg_length, 0.001f, 0.02f);
	ImGui::SliderInt("Hair Segment Number", &hair_seg_num, 1, MAX_SEG_NUM - 1);
	ImGui::SliderFloat("Hair Stiffness", &hair_stiff, 1.0f, 10.0f);
	ImGui::SliderFloat("Gravity", &gravity, 0.0f, 2.0f);
	ImGui::SliderFloat("Damping", &damping, 0.0f, 1.0f);

	ImGui::Render();
}

void Display(void)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (float)(wWindow) / hWindow, 0.01f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(cam_d * sinf(cam_h), cam_v, cam_d * cosf(cam_h), // eye
	          0.0f, 0.0f, 0.0f, // center
		      0.0f, 1.0f, 0.0f);// up

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all

	draw_box();
	//compute_hair();

	glDisable(GL_DEPTH_TEST);
	for (int i = 0; i < sphere_tri_num; i++)
	{
		for (int j = 0; j < tess_num; j++)
		{
			glColor3f(1.0f, 0.0f, 1.0f);
			glBegin(GL_LINE_STRIP);
			for (int k = 0; k < hair_seg_num; k++)
			{
				glVertex3fv((GLfloat*)&pos[i][j][k]);
			}
			glEnd();
			glColor3f(1.0f, 1.0f, 1.0f);
			for (int k = 0; k < hair_seg_num; k++)
			{
				glBegin(GL_POINTS);
				glVertex3fv((GLfloat*)&pos[i][j][k]);
				glEnd();
			}
		}
	}
	glEnable(GL_DEPTH_TEST);

	draw_gui();
	glutSwapBuffers();
	//glutPostRedisplay();
}

void Init(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1000.f);

	glewInit();
	ImGui_ImplGlut_Init();
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0 , GL_POSITION, light_pos);
	glLightf(GL_LIGHT0, GL_AMBIENT, 0.0f);
	glLightf(GL_LIGHT0, GL_DIFFUSE, 0.6f);
	glLightf(GL_LIGHT0, GL_SPECULAR, 0.8f);
	glLightModelf(GL_LIGHT_MODEL_AMBIENT, 0.1f);

	load_obj("sphere.obj", sphere_vertices, sphere_tex_coord, sphere_normals);
	
	glm::vec3 hair_color = glm::vec3(0.0f, 0.0f, 1.0f);
	for (int tri = 0; tri < sphere_tri_num; tri++)
	{
		for (int tess = 0; tess < 3; tess++)
		{
			for (int k = 0; k < 3; k++)
			{
				pos[tri][tess][0][k] = sphere_vertices[3 * tri + tess][k];
				vel[tri][tess][0][k] = 0.0f;
				col[tri][tess][0][k] = hair_color[k];
			}
			for (int seg = 1; seg < MAX_SEG_NUM; seg++)
			{
				for (int k = 0; k < 3; k++)
				{
					pos[tri][tess][seg][k] = pos[tri][tess][seg-1][k] + seg * hair_seg_length * sphere_normals[3 * tri + tess][k];
					vel[tri][tess][seg][k] = 0.0f;
					col[tri][tess][seg][k] = hair_color[k];
				}
			}
		}
	}

	tess_num = (hair_density + 1) * (hair_density + 2) / 2;
	cur_time = clock() / CLK_TCK;
}

__global__ void HairKernel(
	float pos[960 * MAX_TESS * MAX_SEG_NUM][3],
	float vel[960 * MAX_TESS * MAX_SEG_NUM][3],
	float col[960 * MAX_TESS * MAX_SEG_NUM][3],
	const int density, 
	const int tess_num,
	const int seg_num,
	const float seg_len,
	const float stiff,
	const float gravity,
	const float damping,
	const float dt)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if ((i >= 960) || (j >= tess_num)) return;
	// compute the tessellation
	/*int tri_idx = i * MAX_TESS * MAX_SEG_NUM;
	int tess_idx = 2;
	float axis1, axis2;
	for (int a1 = 0; a1 < density; a1++)
	{
		for (int a2 = 0; a2 < density - a1; a2++)
		{
			if ((a1 + a2 > 0) && !(a1 == 0 && a2 == density) && !(a1 == density && a2 == 0))
			{
				tess_idx++;
				if (tess_idx == j)
				{
					axis1 = (float)a1;
					axis2 = (float)a2;
					break;
				}
			}
		}
		if (tess_idx == j)
			break;
	}
	axis1 /= density;
	axis2 /= density;	
	for (int seg = 0; seg < 2; seg++)
	{
		for (int k = 0; k < 3; k++)
		{
			pos[tri_idx + j * MAX_SEG_NUM + seg][k] = pos[tri_idx + seg][k]
				+ axis1 * (pos[tri_idx + MAX_SEG_NUM + seg][k] - pos[tri_idx + seg][k])
				+ axis2 * (pos[tri_idx + 2 * MAX_SEG_NUM + seg][k] - pos[tri_idx + seg][k]);
			vel[tri_idx + j * MAX_SEG_NUM + seg][k] = 0.0f;
			col[tri_idx + j * MAX_SEG_NUM + seg][k] = col[tri_idx + seg][k]
				+ axis1 * (col[tri_idx + MAX_SEG_NUM + seg][k] - col[tri_idx + seg][k])
				+ axis2 * (col[tri_idx + 2 * MAX_SEG_NUM + seg][k] - col[tri_idx + seg][k]);
		}
	}*/
	// compute the segments
	float cur_len, f_spring[3], v[3];
	int idx;
	for (int s = 1; s < seg_num; s++)
	{
		idx = i * MAX_TESS * MAX_SEG_NUM + j * MAX_SEG_NUM + s;
		cur_len = sqrtf((pos[idx][0] - pos[idx - 1][0]) * (pos[idx][0] - pos[idx - 1][0]) +
			(pos[idx][1] - pos[idx - 1][1]) * (pos[idx][1] - pos[idx - 1][1]) +
			(pos[idx][2] - pos[idx - 1][2]) * (pos[idx][2] - pos[idx - 1][2])) + 0.000001;

		f_spring[0] = stiff * (seg_len - cur_len) * (pos[idx][0] - pos[idx - 1][0]) / cur_len;
		f_spring[1] = stiff * (seg_len - cur_len) * (pos[idx][1] - pos[idx - 1][1]) / cur_len;
		f_spring[2] = stiff * (seg_len - cur_len) * (pos[idx][2] - pos[idx - 1][2]) / cur_len;
		v[0] = vel[idx][0] + f_spring[0] * dt;
		v[1] = vel[idx][1] + (f_spring[1] + -1.0f * gravity) * dt;
		v[2] = vel[idx][2] + f_spring[2] * dt;
		v[0] *= damping;
		v[1] *= damping;
		v[2] *= damping;

		pos[idx][0] = pos[idx][0] + v[0];
		pos[idx][1] = pos[idx][1] + v[1];
		pos[idx][2] = pos[idx][2] + v[2];
		vel[idx][0] = v[0];
		vel[idx][1] = v[1];
		vel[idx][2] = v[2];
		col[idx][0] = col[idx - 1][0];
		col[idx][1] = col[idx - 1][1];
		col[idx][2] = col[idx - 1][2];
	}
}

void compute_hair()
{
	cudaError_t error;
	int size_info = 960 * MAX_TESS * MAX_SEG_NUM * 3 * sizeof(float);
	// allocate space
	error = cudaMalloc((void**)&d_pos, size_info);
	if (error != cudaSuccess) Cleanup(false);
	error = cudaMalloc((void**)&d_vel, size_info);
	if (error != cudaSuccess) Cleanup(false);
	error = cudaMalloc((void**)&d_col, size_info);
	if (error != cudaSuccess) Cleanup(false);
	// copy data
	error = cudaMemcpy(d_pos, pos, size_info, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) Cleanup(false);
	error = cudaMemcpy(d_vel, vel, size_info, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) Cleanup(false);
	error = cudaMemcpy(d_col, col, size_info, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) Cleanup(false);

	//prepare blocks and grid
	const int BLOCKSIZE = 16;
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)sphere_tri_num / dimBlock.x),
		ceil((float)MAX_TESS / dimBlock.y));
	// Invoke kernel
	HairKernel <<<dimGrid, dimBlock >>> (
		(float(*)[3])d_pos,
		(float(*)[3])d_vel,
		(float(*)[3])d_col,
		hair_density, 
		tess_num,
		hair_seg_num,
		hair_seg_length,
		hair_stiff,
		gravity,
		damping,
		d_time);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong: %i\n", error);
	/*error = cudaThreadSynchronize();
	if (error != cudaSuccess) { printf("synchronization is wrong\n"); Cleanup(false); }*/
	// Copy result from device memory to host memory
	int size_hair = sphere_tri_num * (MAX_DENSITY + 1) * (MAX_DENSITY + 2) * MAX_SEG_NUM * 9 * sizeof(float) / 2;
	error = cudaMemcpy(pos, d_pos, size_info, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) { printf("could not copy from device\n"); Cleanup(false); }
	error = cudaMemcpy(vel, d_vel, size_info, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) { printf("could not copy from device\n"); Cleanup(false); }
	error = cudaMemcpy(col, d_col, size_info, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) { printf("could not copy from device\n"); Cleanup(false); }

	Cleanup(true);
}

void Idle(void)
{
	cur_time = clock() / CLK_TCK;
	d_time = (cur_time - old_time) / 10;
	old_time = cur_time;
	compute_hair();
	glutPostRedisplay();
}

void myReshape(int w, int h)
{
	glViewport(0, 0, w, h);
	wWindow = w;
	hWindow = h;
}

void Key(unsigned char key, GLint i, GLint j)
{
	switch (key)
	{
	case 'a':
	case 'A':
		cam_h -= 0.1f;
		if (cam_h < -360.0f)
			cam_h += 360.0f;
		break;
	case 'd':
	case 'D':
		cam_h += 0.1f;
		if (cam_h > 360.0f)
			cam_h -= 360.0f;
		break;
	case 'w':
	case 'W':
		cam_v += 0.1f;
		break;
	case 's':
	case 'S':
		cam_v -= 0.1f;
		break;
	case 'q':
	case 'Q':
		cam_d -= 0.1f;
		if (cam_d < 0.2f)
			cam_d = 0.2f;
		break;
	case 'e':
	case 'E':
		cam_d += 0.1f;
		break;
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y) {
	ImGui_ImplGlut_MouseButtonCallback(button, state);
	/*if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		trackball.Set(true, x, y);
		mouseLeft = true;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP)
	{
		trackball.Set(false, x, y);
		mouseLeft = false;
	}*/
}

void MouseMotion(int x, int y) {
	ImGui_ImplGlut_MouseMotionCallback(x, y);
	glutPostRedisplay();
}

// Host code
int main(int argc, char** argv)
{
	glutInitWindowSize(wWindow, hWindow);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutCreateWindow("Hair Simulation");
	Init();
	glutDisplayFunc(Display);
	glutIdleFunc(Idle);
	glutKeyboardFunc(Key);
	glutReshapeFunc(myReshape);
	glutMouseFunc(Mouse);
	glutMotionFunc(MouseMotion);
	glutMainLoop();
	return 0;
}
