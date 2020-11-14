#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <vector>			//Standard template library classTESS_NUM
#include <iostream>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>

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
const int TRIANGLE_NUM = 960;
const int DENSITY = 3;// 8;
const int MAX_SEG_NUM = 80;
const int TESS_NUM = 10;// 45;
const float HAIR_DIA = 0.001f;

float hair_seg_length = 0.01f;
int hair_seg_num = MAX_SEG_NUM;
float hair_stiff = 20.0f;
float gravity = 0.03f;
float damping = 0.002f;
float wind[3] = { 0.0f, 0.0f, 0.0f };

bool sphere_force = false;
float sphere_center[3] = { 0.0f, 0.0f, 0.0f };
float sphere_v[3] = { 0.0f, 0.0f, 0.0f };
float const sphere_weight = 20.0f;

std::vector<glm::vec3> sphere_vertices;
std::vector<glm::vec2> sphere_tex_coord;
std::vector<glm::vec3> sphere_normals;
int sphere_tri_num = 0;
int tess_num;

float old_time, cur_time, delta_time;

// CUDA stuff
cudaError error;
void *d_pos, *d_dir, *d_vel, *d_col;
float *d_init_pos;
__constant__ float d_sphere_center[3];
__constant__ float d_wind[3];
__constant__ float d_cur_time[1];

// vbo variables
GLuint vbo_pos, vbo_dir, vbo_vel, vbo_col, vbo_idx;
struct cudaGraphicsResource *vbo_res[4];
size_t pos_size, dir_size, vel_size, col_size;

// hair data
GLfloat pos[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3];
GLfloat dir[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3];
GLfloat vel[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3];
GLfloat col[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3];
GLint vert_idx[TRIANGLE_NUM * TESS_NUM * (MAX_SEG_NUM + 1)];


GLfloat init_pos[TRIANGLE_NUM][TESS_NUM][3];
float axis[TESS_NUM][2];

void compute_hair();

void Cleanup(bool noError)
{
	cudaError_t error;
	// Free device memory
	if (d_init_pos) error = cudaFree(d_init_pos);
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

void createVBO(GLfloat data[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3], GLuint *vbo, struct cudaGraphicsResource **vbo_res, void** d_ptr, unsigned int vbo_res_flags) {
	cudaError_t error;

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer objects
	glBufferData(GL_ARRAY_BUFFER, TRIANGLE_NUM * TESS_NUM * MAX_SEG_NUM * 3 * sizeof(float), data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	error = cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
	if (error != cudaSuccess) printf("Something went wrong (RegisterBuffer): %s\n", cudaGetErrorString(error));

	size_t size;
	error = cudaGraphicsMapResources(1, vbo_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong (MapResources): %i\n", error);
	error = cudaGraphicsResourceGetMappedPointer(d_ptr, &size, *vbo_res);
	if (error != cudaSuccess) printf("Something went wrong (GetMappedPointer): %i\n", error);
	error = cudaGraphicsUnmapResources(1, vbo_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong (UnmapResources): %i\n", error);
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

	//ImGui::SliderFloat("Hair Segment Length", &hair_seg_length, 0.001f, 0.01f);
	ImGui::SliderFloat("Hair Stiffness", &hair_stiff, 5.0f, 24.0f);
	ImGui::SliderFloat("Gravity", &gravity, 0.0f, 0.1f);
	ImGui::SliderFloat("Damping", &damping, 0.0f, 0.005f);
	ImGui::SliderFloat3("Wind", wind, -0.2f, 0.2f);
	ImGui::Text("WASDQE: Camera Control     "); ImGui::SameLine(); ImGui::Text("F: Enable/Disable Forces on the Sphere  ");
	ImGui::Text("IJKLUO: Sphere Control     "); ImGui::SameLine(); ImGui::Text("Space: Sphere Jump (When Forces Enabled)");
	ImGui::Text("T: Quit                    "); ImGui::SameLine(); ImGui::Text("R: Reset Wind");

	ImGui::Render();
}

void Display(void)
{}

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

	// weights for tessellation
	int count = 0;
	for (int i = 0; i <= DENSITY; i++)
	{
		for (int j = 0; j <= DENSITY - i; j++)
		{
			axis[count][0] = (float)i / (float)DENSITY;
			axis[count][1] = (float)j / (float)DENSITY;
			count++;
		}
	}
	
	glm::vec3 hair_color = glm::vec3(0.0f, 0.0f, 1.0f);
	float dir_len;
	for (int tri = 0; tri < sphere_tri_num; tri++)
	{
		for (int tess = 0; tess < TESS_NUM; tess++)
		{
			for (int k = 0; k < 3; k++)
			{
				pos[tri][tess][0][k] = sphere_vertices[3 * tri][k]
					+ axis[tess][0] * (sphere_vertices[3 * tri + 1][k] - sphere_vertices[3 * tri][k])
					+ axis[tess][1] * (sphere_vertices[3 * tri + 2][k] - sphere_vertices[3 * tri][k]);
				dir[tri][tess][0][k] = (sphere_normals[3 * tri][k]
					+ axis[tess][0] * (sphere_normals[3 * tri + 1][k] - sphere_normals[3 * tri][k])
					+ axis[tess][1] * (sphere_normals[3 * tri + 2][k] - sphere_normals[3 * tri][k]));
				vel[tri][tess][0][k] = 0.0f;
				col[tri][tess][0][k] = hair_color[k];
				init_pos[tri][tess][k] = pos[tri][tess][0][k];
			}
			dir_len = sqrt(dir[tri][tess][0][0] * dir[tri][tess][0][0] +
				dir[tri][tess][0][1] * dir[tri][tess][0][1] +
				dir[tri][tess][0][2] * dir[tri][tess][0][2]);
			dir[tri][tess][0][0] /= dir_len;
			dir[tri][tess][0][1] /= dir_len;
			dir[tri][tess][0][2] /= dir_len;
			for (int seg = 1; seg < MAX_SEG_NUM; seg++)
			{
				for (int k = 0; k < 3; k++)
				{
					pos[tri][tess][seg][k] = pos[tri][tess][seg-1][k] + hair_seg_length * dir[tri][tess][0][k];
					dir[tri][tess][seg][k] = dir[tri][tess][0][k];
					vel[tri][tess][seg][k] = 0.0f;
					col[tri][tess][seg][k] = col[tri][tess][seg - 1][k] + (1.0f - hair_color[k]) / MAX_SEG_NUM;
				}
			}
		}
	}

	// create vbo
	createVBO(pos, &vbo_pos, &vbo_res[0], &d_pos, cudaGraphicsMapFlagsNone);
	createVBO(dir, &vbo_dir, &vbo_res[1], &d_dir, cudaGraphicsMapFlagsNone);
	createVBO(vel, &vbo_vel, &vbo_res[2], &d_vel, cudaGraphicsMapFlagsNone);
	createVBO(col, &vbo_col, &vbo_res[3], &d_col, cudaGraphicsMapFlagsNone);

	// indices vbo
	count = 0;
	for (int i = 0; i < TRIANGLE_NUM; i++)
	{
		for (int j = 0; j < TESS_NUM; j++)
		{
			for (int k = 0; k < MAX_SEG_NUM; k++)
			{
				vert_idx[count] = i * TESS_NUM * MAX_SEG_NUM + j * MAX_SEG_NUM + k;
				count++;
			}
			vert_idx[count] = -1;
			count++;
		}
	}
	glGenBuffers(1, &vbo_idx);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_idx);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLshort) * TRIANGLE_NUM * TESS_NUM * (MAX_SEG_NUM + 1), vert_idx, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	cudaError error;

	error = cudaMalloc((void**)&d_init_pos, TRIANGLE_NUM * TESS_NUM * 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not allocate on device (sphere_init_pos)\n"); Cleanup(false); }
	error = cudaMemcpy(d_init_pos, init_pos, TRIANGLE_NUM * TESS_NUM * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) { printf("could not copy to device (sphere_init_pos)\n"); Cleanup(false); }
}

__global__ void HairKernel(
	float pos[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3],
	float dir[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3],
	float vel[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3],
	float col[TRIANGLE_NUM][TESS_NUM][MAX_SEG_NUM][3],
	float init_pos[TRIANGLE_NUM][TESS_NUM][3],
	float hair_seg_length,
	int hair_seg_num,
	float hair_stiff,
	float gravity,
	float damping,
	float dt,
	float box_size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if ((i >= TRIANGLE_NUM) || (j >= TESS_NUM)) return;
	
	// sphere position
	pos[i][j][0][0] = d_sphere_center[0] + init_pos[i][j][0];
	pos[i][j][0][1] = d_sphere_center[1] + init_pos[i][j][1];
	pos[i][j][0][2] = d_sphere_center[2] + init_pos[i][j][2];

	// compute the segments
	float theo_pos[3], f_spring[3], v[3] = { 0.0f, 0.0f, 0.0f }, cur_len, theo_len, dir_len;
	for (int s = 1; s < hair_seg_num; s++)
	{
		cur_len = sqrt((pos[i][j][s][0] - pos[i][j][s - 1][0]) * (pos[i][j][s][0] - pos[i][j][s - 1][0]) +
			(pos[i][j][s][1] - pos[i][j][s - 1][1]) * (pos[i][j][s][1] - pos[i][j][s - 1][1]) +
			(pos[i][j][s][2] - pos[i][j][s - 1][2]) * (pos[i][j][s][2] - pos[i][j][s - 1][2]));
		theo_pos[0] = pos[i][j][s - 1][0] + hair_seg_length * dir[i][j][s - 1][0];
		theo_pos[1] = pos[i][j][s - 1][1] + hair_seg_length * dir[i][j][s - 1][1];
		theo_pos[2] = pos[i][j][s - 1][2] + hair_seg_length * dir[i][j][s - 1][2];
		f_spring[0] = 10.0f * (theo_pos[0] - pos[i][j][s][0]) + (hair_seg_length - cur_len) * (pos[i][j][s][0] - pos[i][j][s - 1][0]);
		f_spring[1] = 10.0f * (theo_pos[1] - pos[i][j][s][1]) + (hair_seg_length - cur_len) * (pos[i][j][s][1] - pos[i][j][s - 1][1]);
		f_spring[2] = 10.0f * (theo_pos[2] - pos[i][j][s][2]) + (hair_seg_length - cur_len) * (pos[i][j][s][2] - pos[i][j][s - 1][2]);

		// bending
		if (s < hair_seg_num - 1)
		{
			f_spring[0] += (pos[i][j][s - 1][0] - pos[i][j][s][0]) + (pos[i][j][s + 1][0] - pos[i][j][s][0]);
			f_spring[1] += (pos[i][j][s - 1][1] - pos[i][j][s][1]) + (pos[i][j][s + 1][1] - pos[i][j][s][1]);
			f_spring[2] += (pos[i][j][s - 1][2] - pos[i][j][s][2]) + (pos[i][j][s + 1][2] - pos[i][j][s][2]);
		}

		v[0] = vel[i][j][s][0] * damping;
		v[1] = vel[i][j][s][1] * damping;
		v[2] = vel[i][j][s][2] * damping;
		v[0] += (hair_stiff * f_spring[0] + abs(cosf(d_cur_time[0] + s - j)) * d_wind[0]) * dt;
		v[1] += (hair_stiff * f_spring[1] - gravity + abs(sinf(d_cur_time[0] - s - j)) * d_wind[1]) * dt;
		v[2] += (hair_stiff * f_spring[2] + abs(sinf(d_cur_time[0] + s + i)) * d_wind[2]) * dt;
		
		v[0] *= damping;
		v[1] *= damping;
		v[2] *= damping;

		theo_pos[0] = pos[i][j][s][0] + v[0];
		theo_pos[1] = pos[i][j][s][1] + v[1];
		theo_pos[2] = pos[i][j][s][2] + v[2];
		theo_len = sqrt((theo_pos[0] - pos[i][j][s - 1][0]) * (theo_pos[0] - pos[i][j][s - 1][0]) + 
			(theo_pos[1] - pos[i][j][s - 1][1]) * (theo_pos[1] - pos[i][j][s - 1][1]) + 
			(theo_pos[2] - pos[i][j][s - 1][2]) * (theo_pos[2] - pos[i][j][s - 1][2]));

		pos[i][j][s][0] = pos[i][j][s - 1][0] + hair_seg_length * (theo_pos[0] - pos[i][j][s - 1][0]) / theo_len;
		pos[i][j][s][1] = pos[i][j][s - 1][1] + hair_seg_length * (theo_pos[1] - pos[i][j][s - 1][1]) / theo_len;
		pos[i][j][s][2] = pos[i][j][s - 1][2] + hair_seg_length * (theo_pos[2] - pos[i][j][s - 1][2]) / theo_len;

		vel[i][j][s][0] = v[0];
		vel[i][j][s][1] = v[1];
		vel[i][j][s][2] = v[2];

		// collision with the sphere
		if ((pos[i][j][s][0] - d_sphere_center[0]) * (pos[i][j][s][0] - d_sphere_center[0]) +
			(pos[i][j][s][1] - d_sphere_center[1]) * (pos[i][j][s][1] - d_sphere_center[1]) +
			(pos[i][j][s][2] - d_sphere_center[2]) * (pos[i][j][s][2] - d_sphere_center[2]) <= 1.0f)
		{
			pos[i][j][s][0] = d_sphere_center[0] + 1.00001f * (pos[i][j][s][0] - d_sphere_center[0]) /
				sqrtf((pos[i][j][s][0] - d_sphere_center[0]) * (pos[i][j][s][0] - d_sphere_center[0]) +
				(pos[i][j][s][1] - d_sphere_center[1]) * (pos[i][j][s][1] - d_sphere_center[1]) +
				(pos[i][j][s][2] - d_sphere_center[2]) * (pos[i][j][s][2] - d_sphere_center[2]) + 0.000001);
			vel[i][j][s][0] += (1.0f + damping) * (vel[i][j][s][0] * (pos[i][j][s][0] - d_sphere_center[0]) +
				vel[i][j][s][1] * (pos[i][j][s][1] - d_sphere_center[1]) +
				vel[i][j][s][2] * (pos[i][j][s][2] - d_sphere_center[2])) * (pos[i][j][s][0] - d_sphere_center[0]);
			pos[i][j][s][1] = d_sphere_center[1] + 1.00001f * (pos[i][j][s][1] - d_sphere_center[1]) /
				sqrtf((pos[i][j][s][0] - d_sphere_center[0]) * (pos[i][j][s][0] - d_sphere_center[0]) +
				(pos[i][j][s][1] - d_sphere_center[1]) * (pos[i][j][s][1] - d_sphere_center[1]) +
					(pos[i][j][s][2] - d_sphere_center[2]) * (pos[i][j][s][2] - d_sphere_center[2]) + 0.000001);
			vel[i][j][s][1] += (1.0f + damping) * (vel[i][j][s][0] * (pos[i][j][s][0] - d_sphere_center[0]) +
				vel[i][j][s][1] * (pos[i][j][s][1] - d_sphere_center[1]) +
				vel[i][j][s][2] * (pos[i][j][s][2] - d_sphere_center[2])) * (pos[i][j][s][1] - d_sphere_center[1]);
			pos[i][j][s][2] = d_sphere_center[2] + 1.00001f * (pos[i][j][s][2] - d_sphere_center[2]) /
				sqrtf((pos[i][j][s][0] - d_sphere_center[0]) * (pos[i][j][s][0] - d_sphere_center[0]) +
				(pos[i][j][s][1] - d_sphere_center[1]) * (pos[i][j][s][1] - d_sphere_center[1]) +
					(pos[i][j][s][2] - d_sphere_center[2]) * (pos[i][j][s][2] - d_sphere_center[2]) + 0.000001);
			vel[i][j][s][2] += (1.0f + damping) * (vel[i][j][s][0] * (pos[i][j][s][0] - d_sphere_center[0]) +
				vel[i][j][s][1] * (pos[i][j][s][1] - d_sphere_center[1]) +
				vel[i][j][s][2] * (pos[i][j][s][2] - d_sphere_center[2])) * (pos[i][j][s][2] - d_sphere_center[2]);
		}

		// collide with the box
		if (pos[i][j][s][0] >= box_size)
		{
			pos[i][j][s][0] = box_size;
			vel[i][j][s][0] = -1.0f * damping * abs(vel[i][j][s][0]);
		}
		else if (pos[i][j][s][0] <= -1.0f * box_size)
		{
			pos[i][j][s][0] = -1.0f * box_size;
			vel[i][j][s][0] = damping * abs(vel[i][j][s][0]);
		}
		if (pos[i][j][s][1] >= box_size)
		{
			pos[i][j][s][1] = box_size;
			vel[i][j][s][1] = -1.0f * damping * abs(vel[i][j][s][1]);
		}
		else if (pos[i][j][s][1] <= -1.0f * box_size)
		{
			pos[i][j][s][1] = -1.0f * box_size;
			vel[i][j][s][1] = damping * abs(vel[i][j][s][1]);
		}
		if (pos[i][j][s][2] >= box_size)
		{
			pos[i][j][s][2] = box_size;
			vel[i][j][s][2] = -1.0f * damping * abs(vel[i][j][s][2]);
		}
		else if (pos[i][j][s][2] <= -1.0f * box_size)
		{
			pos[i][j][s][2] = -1.0f * box_size;
			vel[i][j][s][2] = damping * abs(vel[i][j][s][2]);
		}

		dir[i][j][s][0] = pos[i][j][s][0] - pos[i][j][s - 1][0];
		dir[i][j][s][1] = pos[i][j][s][1] - pos[i][j][s - 1][1];
		dir[i][j][s][2] = pos[i][j][s][2] - pos[i][j][s - 1][2];
		dir_len = sqrt(dir[i][j][s][0] * dir[i][j][s][0] + dir[i][j][s][1] * dir[i][j][s][1] + dir[i][j][s][2] * dir[i][j][s][2]) + 0.000001;
		dir[i][j][s][0] /= dir_len;
		dir[i][j][s][1] /= dir_len;
		dir[i][j][s][2] /= dir_len;
		/*col[i][j][s][0] = col[i][j][s - 1][0];
		col[i][j][s][1] = col[i][j][s - 1][1];
		col[i][j][s][2] = col[i][j][s - 1][2];*/
	}
}

void compute_hair()
{
	cudaError_t error;

	//prepare blocks and grid
	const int BLOCKSIZE = 16;
	dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid(ceil((float)sphere_tri_num / dimBlock.x),
		ceil((float)TESS_NUM / dimBlock.y));

	// update sphere center and wind
	error = cudaMemcpyToSymbol(d_sphere_center, sphere_center, 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (sphere_center)\n"); Cleanup(false); }
	error = cudaMemcpyToSymbol(d_wind, wind, 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (wind)\n"); Cleanup(false); }
	error = cudaMemcpyToSymbol(d_cur_time, &cur_time, sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (cur_time)\n"); Cleanup(false); }

	// Invoke kernel
	error = cudaGraphicsMapResources(4, vbo_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong when mapping vbo: %i\n", error);
	HairKernel << <dimGrid, dimBlock >> > (
		(float(*)[TESS_NUM][MAX_SEG_NUM][3]) d_pos,
		(float(*)[TESS_NUM][MAX_SEG_NUM][3]) d_dir,
		(float(*)[TESS_NUM][MAX_SEG_NUM][3]) d_vel,
		(float(*)[TESS_NUM][MAX_SEG_NUM][3]) d_col,
		(float(*)[TESS_NUM][3]) d_init_pos,
		hair_seg_length,
		hair_seg_num,
		hair_stiff,
		gravity,
		damping,
		delta_time,
		box_size);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong when executing kernel: %i\n", error);
	cudaGraphicsUnmapResources(4, vbo_res, NULL);//sync!
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong when unmapping vbo: %i\n", error);
}

void Idle(void)
{
	cur_time = clock() / CLK_TCK;
	delta_time = 2;//(cur_time - old_time);
	old_time = cur_time;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (float)(wWindow) / hWindow, 0.01f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(cam_d * sinf(cam_h), cam_v, cam_d * cosf(cam_h), // eye
		0.0f, cam_v, 0.0f, // center
		0.0f, 1.0f, 0.0f);// up

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all

	draw_box();

	if (sphere_force)
	{
		sphere_v[0] += wind[0] * delta_time / sphere_weight;
		sphere_v[1] += (-1.0f * gravity + wind[1]) * delta_time / sphere_weight;
		sphere_v[2] += wind[2] * delta_time / sphere_weight;

		sphere_center[0] += sphere_v[0];
		sphere_center[1] += sphere_v[1];
		sphere_center[2] += sphere_v[2];

		sphere_v[0] *= 0.9f;
		sphere_v[1] *= 0.9f;
		sphere_v[2] *= 0.9f;

		if (sphere_center[0] > box_size - 1.0f)
		{
			sphere_center[0] = box_size - 1.0f;
			sphere_v[0] = -1.0f * abs(sphere_v[0]);
		}
		else if (sphere_center[0] < -1.0f *  box_size + 1.0f)
		{
			sphere_center[0] = -1.0f *  box_size + 1.0f;
			sphere_v[0] = abs(sphere_v[0]);
		}
		if (sphere_center[1] > box_size - 1.0f)
		{
			sphere_center[1] = box_size - 1.0f;
			sphere_v[1] = -1.0f * abs(sphere_v[1]);
		}
		else if (sphere_center[1] < -1.0f * box_size + 1.0f)
		{
			sphere_center[1] = -1.0f *  box_size + 1.0f;
			sphere_v[1] = abs(sphere_v[1]);
		}
		if (sphere_center[2] > box_size - 1.0f)
		{
			sphere_center[2] = box_size - 1.0f;
			sphere_v[2] = -1.0f * abs(sphere_v[2]);
		}
		else if (sphere_center[2] < -1.0f *  box_size + 1.0f)
		{
			sphere_center[2] = -1.0f *  box_size + 1.0f;
			sphere_v[2] = abs(sphere_v[2]);
		}
	}
	else
	{
		sphere_v[0]= 0;
		sphere_v[1]= 0;
		sphere_v[2]= 0;
	}
	
	compute_hair();

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(-1);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_idx);
	glEnableClientState(GL_INDEX_ARRAY);
	glIndexPointer(GL_INT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, 0);

	//glTexCoordPointer(2, GL_FLOAT, 0, 0);

	glDrawElements(GL_LINE_STRIP, TRIANGLE_NUM * TESS_NUM * (MAX_SEG_NUM + 1), GL_UNSIGNED_INT, 0);

	glDisableClientState(GL_INDEX_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisable(GL_PRIMITIVE_RESTART);
	glEnable(GL_DEPTH_TEST);

	draw_gui();
	glutSwapBuffers();
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
		// camera
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
		// sphere
	case 'i':
	case 'I':
		if (sphere_center[1] < box_size - 1.0f)
		{
			sphere_center[1] += 0.1f;
		}
		break;
	case 'k':
	case 'K':
		if (sphere_center[1] > -1.0f * box_size + 1.0f)
		{
			sphere_center[1] -= 0.1f;
		}
		break;
	case 'j':
	case 'J':
		if (sphere_center[0] > -1.0f * box_size + 1.0f)
		{
			sphere_center[0] -= 0.1f;
		}
		break;
	case 'l':
	case 'L':
		if (sphere_center[0] < box_size - 1.0f)
		{
			sphere_center[0] += 0.1f;
		}
		break;
	case 'u':
	case 'U':
		if (sphere_center[2] > -1.0f * box_size + 1.0f)
		{
			sphere_center[2] -= 0.1f;
		}
		break;
	case 'o':
	case 'O':
		if (sphere_center[2] < box_size - 1.0f)
		{
			sphere_center[2] += 0.1f;
		}
		break;
		// sphere force on/off
	case 'f':
	case 'F':
		sphere_force = !sphere_force;
		printf("sphere force: %d\n", sphere_force);
		break;
		// sphere jump
	case ' ':
		sphere_v[1] += gravity * sphere_weight;
		break;
	case 'r':
	case'R':
		wind[0] = 0.0f;
		wind[1] = 0.0f;
		wind[2] = 0.0f;
		break;
	case 't':
	case 'T':
		Cleanup(true);
		cudaDeviceReset();
		exit(0);
	}
	glutPostRedisplay();
}

void Mouse(int button, int state, int x, int y) {
	ImGui_ImplGlut_MouseButtonCallback(button, state);
	glutPostRedisplay();
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
