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

const int WIN_W = 1200, WIN_H = 800;

GLint wWindow = 1200;
GLint hWindow = 800;

GLfloat light_pos[3] = { 3.0f, 6.0f, 0.0f };
const float box_size = 4.0f;
GLfloat box_color_a[4] = { 0.1f, 0.2f, 0.2f, 1.0f };
GLfloat box_color_d[4] = { 0.3f, 0.5f, 0.5f, 1.0f };

// camera ctrl
float cam_h = 0.0f;
float cam_v = 0.0f;
float cam_d = 2.0f * box_size;

bool mode = false;

// param
const int TRIANGLE_NUM = 960;
const int DENSITY = 3;
const int SEG_NUM = 50;
const float SEG_LEN = 0.01f;
const int TESS_NUM = 10;
const float DENS_CELL_SIZE = 0.004;
const int DENS_GRID_SIZE = 800; //(SEG_NUM * SEG_LEN * 2 + 2.0f) / DENS_CELL_SIZE

//float hair_seg_length = 0.01f;
int hair_seg_num = SEG_NUM;
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

float old_time, cur_time, delta_time, time_diff;
float inv_VP[16];

// CUDA stuff
cudaError error;
void *d_pos, *d_dir, *d_vel, *d_col, *d_pix;
float *d_init_pos;
int *d_dense;
__constant__ float d_sphere_center[3];
__constant__ float d_wind[3];
__constant__ float d_cur_time[1];
__constant__ float d_inv_VP[16];
__constant__ float d_light_pos[3];

// vbo variables
GLuint vbo_pos, vbo_dir, vbo_vel, vbo_col, vbo_idx;
struct cudaGraphicsResource *vbo_res[4];
size_t pos_size, dir_size, vel_size, col_size;
// pixel buffer
GLuint pix_buffer;
cudaGraphicsResource *pixel_res;

// hair data
GLfloat pos[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3];
GLfloat dir[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3];
GLfloat vel[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3];
GLfloat col[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3];
GLint vert_idx[TRIANGLE_NUM * TESS_NUM * (SEG_NUM + 1)];

GLfloat init_pos[TRIANGLE_NUM][TESS_NUM][3];
float axis[TESS_NUM][2];

void compute_hair();

void Cleanup(bool noError)
{
	cudaError_t error;
	// Free device memory
	if (d_init_pos) error = cudaFree(d_init_pos);
	if (!noError || error != cudaSuccess) printf("Something failed \n");
	if (d_dense) error = cudaFree(d_dense);
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

void createVBO(GLfloat data[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3], GLuint *vbo, struct cudaGraphicsResource **vbo_res, void** d_ptr, unsigned int vbo_res_flags) {
	cudaError_t error;

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	// initialize buffer objects
	glBufferData(GL_ARRAY_BUFFER, TRIANGLE_NUM * TESS_NUM * SEG_NUM * 3 * sizeof(float), data, GL_DYNAMIC_DRAW);
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

void create_pixel_buffer(GLuint *pix_buffer, struct cudaGraphicsResource **pixel_res, void** d_ptr, unsigned int vbo_res_flags, GLuint buffer_type, int channel_num)
{
	cudaError_t error;

	glGenBuffers(1, pix_buffer);
	glBindBuffer(buffer_type, *pix_buffer);
	glBufferData(buffer_type, hWindow * wWindow * channel_num * sizeof(float), NULL, GL_DYNAMIC_DRAW);

	cudaGraphicsGLRegisterBuffer(pixel_res, *pix_buffer, cudaGraphicsMapFlagsNone);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong (register pixel buffer): %s\n", cudaGetErrorString(error));

	size_t size;
	error = cudaGraphicsMapResources(1, pixel_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong (map pixel buffer): %s\n", cudaGetErrorString(error));
	error = cudaGraphicsResourceGetMappedPointer(d_ptr, &size, *pixel_res);
	if (error != cudaSuccess) printf("Something went wrong (GetMappedPointer - pixel): %s\n", cudaGetErrorString(error));
	error = cudaGraphicsUnmapResources(1, pixel_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong (unmap pixel buffer): %s\n", cudaGetErrorString(error));

	glBindBuffer(buffer_type, 0);
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
	//ImGui::SliderFloat("Hair Stiffness", &hair_stiff, 5.0f, 24.0f);
	//ImGui::SliderFloat("Gravity", &gravity, 0.0f, 0.1f);
	//ImGui::SliderFloat("Damping", &damping, 0.0f, 0.005f);
	ImGui::SliderFloat3("Wind", wind, -0.2f, 0.2f);
	ImGui::Text("WASDQE: Camera Control     "); ImGui::SameLine(); ImGui::Text("F: Enable/Disable Forces on the Sphere  ");
	ImGui::Text("IJKLUO: Sphere Control     "); ImGui::SameLine(); ImGui::Text("Space: Sphere Jump (When Forces Enabled)");
	ImGui::Text("M: Switch display mode     "); ImGui::SameLine(); ImGui::Text("R: Reset Wind");
	ImGui::Text("T: Quit                    "); ImGui::SameLine(); ImGui::Text("fps = %f", CLOCKS_PER_SEC / time_diff);

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
				dir[tri][tess][0][k] += (0.36 * sinf(clock()));
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
			for (int seg = 1; seg < SEG_NUM; seg++)
			{
				for (int k = 0; k < 3; k++)
				{
					pos[tri][tess][seg][k] = pos[tri][tess][seg-1][k] + SEG_LEN * dir[tri][tess][0][k];
					dir[tri][tess][seg][k] = dir[tri][tess][0][k];
					vel[tri][tess][seg][k] = 0.0f;
					col[tri][tess][seg][k] = col[tri][tess][seg - 1][k] + (1.0f - hair_color[k]) / SEG_NUM;
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
			for (int k = 0; k < SEG_NUM; k++)
			{
				vert_idx[count] = i * TESS_NUM * SEG_NUM + j * SEG_NUM + k;
				count++;
			}
			vert_idx[count] = -1;
			count++;
		}
	}
	glGenBuffers(1, &vbo_idx);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_idx);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLshort) * TRIANGLE_NUM * TESS_NUM * (SEG_NUM + 1), vert_idx, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	cudaError error;

	error = cudaMalloc((void**)&d_init_pos, TRIANGLE_NUM * TESS_NUM * 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not allocate on device (sphere_init_pos)\n"); Cleanup(false); }
	error = cudaMemcpy(d_init_pos, init_pos, TRIANGLE_NUM * TESS_NUM * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) { printf("could not copy to device (sphere_init_pos)\n"); Cleanup(false); }

	error = cudaMalloc((void**)&d_dense, DENS_GRID_SIZE * DENS_GRID_SIZE * DENS_GRID_SIZE * sizeof(int));
	if (error != cudaSuccess) { printf("could not allocate on device (dense_grid)\n"); Cleanup(false); }
	error = cudaMemset(d_dense, 0, DENS_GRID_SIZE * DENS_GRID_SIZE * DENS_GRID_SIZE * sizeof(int));
	if (error != cudaSuccess) { printf("could not set value on device (dense_grid)\n"); Cleanup(false); }

	error = cudaMemcpyToSymbol(d_light_pos, &light_pos, 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (light_pos)\n"); Cleanup(false); }

	// pixel buffer
	create_pixel_buffer(&pix_buffer, &pixel_res, &d_pix, cudaGraphicsMapFlagsNone, GL_PIXEL_UNPACK_BUFFER, 4);
}

__device__ void cross(float a[3], float b[3], float c[3])
{
	c[0] = a[1] * b[2] - a[2] * b[1];
	c[1] = a[2] * b[0] - a[0] * b[2];
	c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ float len(float *a)
{
	return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__device__ float dist(float *a, float *b)
{
	return sqrtf((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

__device__ float dot(float a[3], float b[3])
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ float sign(float a)
{
	if (a < 0)
		return -1.0f;
	else
		return 1.0f;
}

__global__ void HairKernel(
	float pos[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3],
	float dir[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3],
	float vel[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3],
	float col[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3],
	float init_pos[TRIANGLE_NUM][TESS_NUM][3],
	int dense_grid[DENS_GRID_SIZE][DENS_GRID_SIZE][DENS_GRID_SIZE],
	float hair_seg_length,
	int hair_seg_num,
	float hair_stiff,
	float gravity,
	float damping,
	float dt,
	float box_size,
	int dens_grid_size,
	float dens_cell_size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	int k = threadIdx.z;
	if ((i >= TRIANGLE_NUM) || (j >= TESS_NUM)) return;

	int sm_idx = blockDim.y * blockDim.z * threadIdx.x + blockDim.z * threadIdx.y + threadIdx.z;

	__shared__ float sm_pos[16*16*3], sm_vel[16 * 16 * 3], sm_dir[16 * 16 * 3], sm_theo_pos[16 * 16 * 3];

	// sphere position
	pos[i][j][0][k] = d_sphere_center[k] + init_pos[i][j][k];

	// compute the segments
	float f_spring, v, dist_to_center, theo_len, dir_len;// , cur_len;
	int dens_idx[3], cell[3];
	for (int s = 1; s < hair_seg_num; s++)
	{
		sm_pos[sm_idx] = pos[i][j][s][k];
		sm_vel[sm_idx] = vel[i][j][s][k];
		sm_dir[sm_idx] = dir[i][j][s][k];

		sm_theo_pos[sm_idx] = pos[i][j][s - 1][k] + hair_seg_length * dir[i][j][s - 1][k];
		f_spring = 10.0f * (sm_theo_pos[sm_idx] - sm_pos[sm_idx]);

		// bending
		if (s < hair_seg_num - 1)
		{
			f_spring += (pos[i][j][s - 1][k] - sm_pos[sm_idx]) + (pos[i][j][s + 1][k] - sm_pos[sm_idx]);
		}
		
		sm_vel[sm_idx] *= damping;
		sm_vel[sm_idx] += (hair_stiff * f_spring - gravity * (k == 1) + 
			(abs(cosf(d_cur_time[0] * 240.7 + s * 2.5f + i * 6.4f + 2.5f)) + abs(sinf(500.3f * d_cur_time[0] * i + j * d_cur_time[0] + s))) * d_wind[k]) * dt;
		sm_vel[sm_idx] *= damping;

		//hair collision
		dens_idx[0] = (int)floor((pos[i][j][s][0] - d_sphere_center[0]) / dens_cell_size) + (dens_grid_size + 1) / 2;
		dens_idx[1] = (int)floor((pos[i][j][s][1] - d_sphere_center[1]) / dens_cell_size) + (dens_grid_size + 1) / 2;
		dens_idx[2] = (int)floor((pos[i][j][s][2] - d_sphere_center[2]) / dens_cell_size) + (dens_grid_size + 1) / 2;
		v = abs(sm_vel[sm_idx]);
		if (k == 0)
		{
			cell[0] = dense_grid[dens_idx[0] - 1][dens_idx[1]][dens_idx[2]];
			cell[1] = dense_grid[dens_idx[0]][dens_idx[1]][dens_idx[2]];
			cell[2] = dense_grid[dens_idx[0] + 1][dens_idx[1]][dens_idx[2]];
			if (cell[1] > 1 && cell[1] > cell[0])
			{
				sm_vel[sm_idx] -= 0.00001f * (cell[1] - cell[0]);
			}
			else if (cell[1] > 1 && cell[1] > cell[2])
			{
				sm_vel[sm_idx] += 0.00001f * (cell[1] - cell[2]);
			}
		}
		else if (k == 1)
		{
			cell[0] = dense_grid[dens_idx[0]][dens_idx[1] - 1][dens_idx[2]];
			cell[1] = dense_grid[dens_idx[0]][dens_idx[1]][dens_idx[2]];
			cell[2] = dense_grid[dens_idx[0]][dens_idx[1] + 1][dens_idx[2]];
			if (cell[1] > 1 && cell[1] > cell[0])
			{
				sm_vel[sm_idx] -= 0.00001f * (cell[1] - cell[0]);
			}
			else if (cell[1] > 1 && cell[1] > cell[2])
			{
				sm_vel[sm_idx] += 0.00001f * (cell[1] - cell[2]);
			}
		}
		else
		{
			cell[0] = dense_grid[dens_idx[0]][dens_idx[1]][dens_idx[2] - 1];
			cell[1] = dense_grid[dens_idx[0]][dens_idx[1]][dens_idx[2]];
			cell[2] = dense_grid[dens_idx[0]][dens_idx[1]][dens_idx[2] + 1];
			if (cell[1] > 1 && cell[1] > cell[0])
			{
				sm_vel[sm_idx] -= 0.00001f * (cell[1] - cell[0]);
			}
			else if (cell[1] > 1 && cell[1] > cell[2])
			{
				sm_vel[sm_idx] += 0.00001f * (cell[1] - cell[2]);
			}
		}

		sm_theo_pos[sm_idx] = sm_pos[sm_idx] + sm_vel[sm_idx];

		__syncthreads();
		
		theo_len = dist(&sm_theo_pos[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z], &pos[i][j][s - 1][0]);
		sm_pos[sm_idx] = pos[i][j][s - 1][k] + hair_seg_length * (sm_theo_pos[sm_idx] - pos[i][j][s - 1][k]) / theo_len;

		__syncthreads();
		
		dist_to_center = dist(&sm_pos[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z], d_sphere_center);

		// collision with the sphere
		if (dist_to_center <= 1.0f)
		{
			sm_pos[sm_idx] = (dist_to_center <= 1.0f) * d_sphere_center[k] + 1.00001f * (sm_pos[sm_idx] - d_sphere_center[k]) / (dist_to_center + 0.000001);
		}
		__syncthreads();
		if (dist_to_center <= 1.0f)
		{
			v = (1.0f + damping) *
				(sm_vel[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z] *
					(sm_pos[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z] - d_sphere_center[0]) +
				sm_vel[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + 1] *
					(sm_pos[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + 1] - d_sphere_center[1]) +
				sm_vel[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + 2] *
					(sm_pos[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + 2] - d_sphere_center[2]));
			sm_vel[sm_idx] += v;
		}

		// collide with the box
		if (sm_pos[sm_idx] > box_size + 0.00001f)
		{
			sm_pos[sm_idx] = box_size - 0.000001f;
			sm_vel[sm_idx] = -1.0f * damping * abs(sm_vel[sm_idx]);
		}
		else if (sm_pos[sm_idx] < -1.0f * box_size - 0.0001f)
		{
			sm_pos[sm_idx] = -1.0f * box_size + 0.000001f;
			sm_vel[sm_idx] = damping * abs(sm_vel[sm_idx]);
		}

		sm_dir[sm_idx] = sm_pos[sm_idx] - pos[i][j][s - 1][k];

		__syncthreads();

		dir_len = len(&sm_dir[threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z]) + 0.000001;
		sm_dir[sm_idx] /= dir_len;

		pos[i][j][s][k] = sm_pos[sm_idx];
		vel[i][j][s][k] = sm_vel[sm_idx];
		dir[i][j][s][k] = sm_dir[sm_idx];
	}
}

__global__ void CollisionKernel(
	float pos[TRIANGLE_NUM][TESS_NUM][SEG_NUM][3],
	int dense_grid[DENS_GRID_SIZE][DENS_GRID_SIZE][DENS_GRID_SIZE],
	int grid_size,
	float cell_size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if ((i >= TRIANGLE_NUM * TESS_NUM) || (j >= SEG_NUM - 1)) return;
	int tri_idx = i / TESS_NUM;
	int tess_idx = i % TESS_NUM;
	float seg_start[3] = { pos[tri_idx][tess_idx][j][0], pos[tri_idx][tess_idx][j][1], pos[tri_idx][tess_idx][j][2] };
	float seg_end[3] = { pos[tri_idx][tess_idx][j + 1][0], pos[tri_idx][tess_idx][j + 1][1], pos[tri_idx][tess_idx][j + 1][2] };
	int x = (int)(floor((seg_start[0] - d_sphere_center[0]) / cell_size) + (grid_size + 1) / 2);
	int y = (int)(floor((seg_start[1] - d_sphere_center[1]) / cell_size) + (grid_size + 1) / 2);
	int z = (int)(floor((seg_start[2] - d_sphere_center[2]) / cell_size) + (grid_size + 1) / 2);
	if (x >= 0 && y >= 0 && z >= 0 && x < grid_size && y < grid_size && z < grid_size)
	{
		atomicAdd(&dense_grid[x][y][z], 1);
		if (x > 0)
		{
			atomicAdd(&dense_grid[x - 1][y][z], 1);
		}
		if (x < grid_size - 1)
		{
			atomicAdd(&dense_grid[x + 1][y][z], 1);
		}
		if (y > 0)
		{
			atomicAdd(&dense_grid[x][y - 1][z], 1);
		}
		if (y < grid_size - 1)
		{
			atomicAdd(&dense_grid[x][y + 1][z], 1);
		}
		if (z > 0)
		{
			atomicAdd(&dense_grid[x][y][z - 1], 1);
		}
		if (z < grid_size - 1)
		{
			atomicAdd(&dense_grid[x][y][z + 1], 1);
		}
		x = (int)(floor(((seg_start[0] + seg_end[0]) / 2 - d_sphere_center[0]) / cell_size) + (grid_size + 1) / 2);
		y = (int)(floor(((seg_start[1] + seg_end[1]) / 2 - d_sphere_center[1]) / cell_size) + (grid_size + 1) / 2);
		z = (int)(floor(((seg_start[2] + seg_end[2]) / 2 - d_sphere_center[2]) / cell_size) + (grid_size + 1) / 2);
		atomicAdd(&dense_grid[x][y][z], 1);
		if (x > 0)
		{
			atomicAdd(&dense_grid[x - 1][y][z], 1);
		}
		if (x < grid_size - 1)
		{
			atomicAdd(&dense_grid[x + 1][y][z], 1);
		}
		if (y > 0)
		{
			atomicAdd(&dense_grid[x][y - 1][z], 1);
		}
		if (y < grid_size - 1)
		{
			atomicAdd(&dense_grid[x][y + 1][z], 1);
		}
		if (z > 0)
		{
			atomicAdd(&dense_grid[x][y][z - 1], 1);
		}
		if (z < grid_size - 1)
		{
			atomicAdd(&dense_grid[x][y][z + 1], 1);
		}
	}
}

__device__ float p_to_line_dist(float p[3], float a[3], float b[3])
{
	float a_to_p[3] = { p[0] - a[0], p[1] - a[1] , p[2] - a[2] };
	float a_to_b[3] = { b[0] - a[0], b[1] - a[1] , b[2] - a[2] };
	float cross_product[3];
	cross(a_to_p, a_to_b, cross_product);
	return len(cross_product) / len(a_to_b);
}

__global__ void PixelKernel(
	int dense_grid[DENS_GRID_SIZE][DENS_GRID_SIZE][DENS_GRID_SIZE],
	float buf[WIN_H][WIN_W][4],
	float SEG_LEN,
	int  grid_size,
	float cell_size)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	int j = blockDim.y*blockIdx.y + threadIdx.y;
	if ((i >= WIN_H) || (j >= WIN_W)) return;

	glm::mat4 inv_vp = glm::make_mat4(d_inv_VP);
	glm::vec4 start_4 = inv_vp * glm::vec4(2.0f * (float)j / (float)(WIN_W - 1) - 1.0f, 2.0f * (float)i / (float)(WIN_H - 1) - 1.0f, -1.0f, 1.0f);
	glm::vec3 start_glm(start_4 / start_4.w);
	glm::vec4 end_4 = inv_vp * glm::vec4(2.0f * (float)j / (float)(WIN_W - 1) - 1.0f, 2.0f * (float)i / (float)(WIN_H - 1) - 1.0f, 1.0f, 1.0f);
	glm::vec3 end_glm(end_4 / end_4.w);
	float start[3] = { start_glm.x, start_glm.y, start_glm.z };
	float end[3] = { end_glm.x, end_glm.y, end_glm.z };
	float limit = 1.0f + SEG_NUM * SEG_LEN;
	// too far away from the sphere
	if (p_to_line_dist(d_sphere_center, start, end) > limit)
	{
		return;
	}

	float rayDir[3] = {(end[0] - start[0]) / dist(start, end), (end[1] - start[1]) / dist(start, end), (end[2] - start[2]) / dist(start, end) };
	float stepSize = p_to_line_dist(d_sphere_center, start, end) - limit;
	float cur_pos[3] = { start[0] + stepSize * rayDir[0], start[1] + stepSize * rayDir[1], start[2] + stepSize * rayDir[2] };
	stepSize = cell_size;
	float travel = dist(end, cur_pos);
	int x, y, z;
	float light[3], n[3], diff, light_to_pos, center_to_pos, light_to_center;
	for (int s = 0; travel > 0.0f; s++, cur_pos[0] += rayDir[0] * stepSize, cur_pos[1] += rayDir[1] * stepSize, cur_pos[2] += rayDir[2] * stepSize, travel -= stepSize)
	{
		x = (int)(floor((cur_pos[0] - d_sphere_center[0]) / cell_size) + (grid_size + 1) / 2);
		y = (int)(floor((cur_pos[1] - d_sphere_center[1]) / cell_size) + (grid_size + 1) / 2);
		z = (int)(floor((cur_pos[2] - d_sphere_center[2]) / cell_size) + (grid_size + 1) / 2);
		if (x >= 0 && y >= 0 && z >= 0 && x < grid_size && y < grid_size && z < grid_size)
		{
			if (dense_grid[x][y][z] > 0)
			{
				light_to_pos = dist(d_light_pos, cur_pos);
				light[0] = (d_light_pos[0] - cur_pos[0]) / light_to_pos;
				light[1] = (d_light_pos[1] - cur_pos[1]) / light_to_pos;
				light[2] = (d_light_pos[2] - cur_pos[2]) / light_to_pos;
				center_to_pos = dist(d_sphere_center, cur_pos);
				n[0] = (cur_pos[0] - d_sphere_center[0]) / center_to_pos;
				n[1] = (cur_pos[1] - d_sphere_center[1]) / center_to_pos;
				n[2] = (cur_pos[2] - d_sphere_center[2]) / center_to_pos;

				diff = max(0.0, dot(n, light));

				light_to_center = dist(d_light_pos, d_sphere_center);
				buf[i][j][0] = min(1.0f, 0.38f + 0.43f * diff + 0.05f * (light_to_center - light_to_pos) + 0.1 * (center_to_pos - 1.6f));
				buf[i][j][1] = min(1.0f, 0.3f + 0.3f * diff + 0.05f * (light_to_center - light_to_pos) + 0.1 * (center_to_pos - 1.6f));
				buf[i][j][2] = min(1.0f, 0.2f + 0.2f * diff + 0.03f * (light_to_center - light_to_pos) + 0.1 * (center_to_pos - 1.6f));
				buf[i][j][3] = min(1.0f, buf[i][j][3] + dense_grid[x][y][z] * 0.08f);
				if (buf[i][j][3] >= 1.0f)
					return;
			}
		}
	}
}

void compute_hair()
{
	cudaError_t error;

	//prepare blocks and grid
	const int BLOCKSIZE = 16;
	dim3 dimBlock_hair(BLOCKSIZE, BLOCKSIZE, 3);
	dim3 dimGrid_hair(ceil((float)sphere_tri_num / dimBlock_hair.x), ceil((float)TESS_NUM / dimBlock_hair.y), 1);

	// update sphere center and wind
	error = cudaMemcpyToSymbol(d_sphere_center, sphere_center, 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (sphere_center)\n"); Cleanup(false); }
	error = cudaMemcpyToSymbol(d_wind, wind, 3 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (wind)\n"); Cleanup(false); }
	error = cudaMemcpyToSymbol(d_cur_time, &cur_time, sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (cur_time)\n"); Cleanup(false); }
	// update inverse VP
	error = cudaMemcpyToSymbol(d_inv_VP, &inv_VP, 16 * sizeof(float));
	if (error != cudaSuccess) { printf("could not copy to device (inv_VP)\n"); Cleanup(false); }

	// Invoke kernel
	error = cudaGraphicsMapResources(4, vbo_res, NULL);
	if (error != cudaSuccess) printf("Something went wrong when mapping vbo: %i\n", error);
	HairKernel << <dimGrid_hair, dimBlock_hair >> > (
		(float(*)[TESS_NUM][SEG_NUM][3]) d_pos,
		(float(*)[TESS_NUM][SEG_NUM][3]) d_dir,
		(float(*)[TESS_NUM][SEG_NUM][3]) d_vel,
		(float(*)[TESS_NUM][SEG_NUM][3]) d_col,
		(float(*)[TESS_NUM][3]) d_init_pos,
		(int(*)[DENS_GRID_SIZE][DENS_GRID_SIZE]) d_dense,
		SEG_LEN,
		hair_seg_num,
		hair_stiff,
		gravity,
		damping,
		delta_time,
		box_size,
		DENS_GRID_SIZE,
		DENS_CELL_SIZE);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong when executing hair kernel: %i\n", error);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) printf("Something went wrong when sync kernel: %i\n", error);
	error = cudaMemset(d_dense, 0, DENS_GRID_SIZE * DENS_GRID_SIZE * DENS_GRID_SIZE * sizeof(int));
	if (error != cudaSuccess) { printf("could not clear dense_grid\n"); Cleanup(false); }

	dim3 dimBlock_colli(BLOCKSIZE, BLOCKSIZE);
	dim3 dimGrid_colli(ceil((float)(sphere_tri_num * TESS_NUM) / dimBlock_colli.x), ceil((float)SEG_NUM / dimBlock_colli.y));
	CollisionKernel << <dimGrid_colli, dimBlock_colli >> > (
		(float(*)[TESS_NUM][SEG_NUM][3]) d_pos,
		(int(*)[DENS_GRID_SIZE][DENS_GRID_SIZE]) d_dense,
		DENS_GRID_SIZE,
		DENS_CELL_SIZE);
	error = cudaGetLastError();
	if (error != cudaSuccess) printf("Something went wrong when executing collision kernel: %i\n", error);
	error = cudaDeviceSynchronize();
	if (error != cudaSuccess) printf("Something went wrong when sync kernel: %i\n", error);
	error = cudaGraphicsUnmapResources(4, vbo_res, NULL);//sync!
	if (error != cudaSuccess) printf("Something went wrong when unmapping vbo: %i\n", error);

	if (mode)
	{
		// pixel buffer
		error = cudaMemset(d_pix, 0, WIN_H * WIN_W * 4 * sizeof(float));
		if (error != cudaSuccess) { printf("could not clear pix_buffer\n"); Cleanup(false); }
		error = cudaGraphicsMapResources(1, &pixel_res, NULL);
		if (error != cudaSuccess) printf("Something went wrong when mapping pix: %i\n", error);
		dim3 dimBlock_pix(BLOCKSIZE, BLOCKSIZE);
		dim3 dimGrid_pix(ceil((float)WIN_H / dimBlock_pix.x),
			ceil((float)WIN_W / dimBlock_pix.y));
		PixelKernel << <dimGrid_pix, dimBlock_pix >> > (
			(int(*)[DENS_GRID_SIZE][DENS_GRID_SIZE]) d_dense,
			(float(*)[WIN_W][4]) d_pix,
			SEG_LEN,
			DENS_GRID_SIZE,
			DENS_CELL_SIZE);
		error = cudaGetLastError();
		if (error != cudaSuccess) printf("Something went wrong when executing pixel kernel: %i\n", error);
		error = cudaDeviceSynchronize();
		if (error != cudaSuccess) printf("Something went wrong when sync kernel: %i\n", error);
		error = cudaGraphicsUnmapResources(1, &pixel_res, NULL);//sync!
		if (error != cudaSuccess) printf("Something went wrong when unmapping pix: %i\n", error);
	}
}

void Idle(void)
{
	cur_time = clock();
	delta_time = 2;
	time_diff = cur_time - old_time;
	old_time = cur_time;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0f, (float)(wWindow) / hWindow, 0.01f, 20.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(cam_d * sinf(cam_h), cam_v, cam_d * cosf(cam_h), // eye
		0.0f, cam_v, 0.0f, // center
		0.0f, 1.0f, 0.0f);// up

	glm::mat4 V = glm::lookAt(glm::vec3(cam_d * sinf(cam_h), cam_v, cam_d * cosf(cam_h)),
		glm::vec3(0.0f, cam_v, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 P = glm::perspective(60.0f, (float)(wWindow) / hWindow, 0.01f, 20.0f);
	glm::mat4 inv_VP_glm = glm::inverse(P * V);
	memcpy(inv_VP, &inv_VP_glm, 16 * sizeof(float));

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

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear all
	draw_box();

	if (mode)
	{
		glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBlendEquation(GL_FUNC_ADD);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pix_buffer);
		glDrawPixels(wWindow, hWindow, GL_RGBA, GL_FLOAT, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glDisable(GL_BLEND);
	}
	else
	{
		glEnable(GL_DEPTH_TEST);
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

		glDrawElements(GL_LINE_STRIP, TRIANGLE_NUM * TESS_NUM * (SEG_NUM + 1), GL_UNSIGNED_INT, 0);

		glDisableClientState(GL_INDEX_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDisable(GL_PRIMITIVE_RESTART);
	}

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
			sphere_center[1] += 0.05f;
		}
		break;
	case 'k':
	case 'K':
		if (sphere_center[1] > -1.0f * box_size + 1.0f)
		{
			sphere_center[1] -= 0.05f;
		}
		break;
	case 'j':
	case 'J':
		if (sphere_center[0] > -1.0f * box_size + 1.0f)
		{
			sphere_center[0] -= 0.05f;
		}
		break;
	case 'l':
	case 'L':
		if (sphere_center[0] < box_size - 1.0f)
		{
			sphere_center[0] += 0.05f;
		}
		break;
	case 'u':
	case 'U':
		if (sphere_center[2] > -1.0f * box_size + 1.0f)
		{
			sphere_center[2] -= 0.05f;
		}
		break;
	case 'o':
	case 'O':
		if (sphere_center[2] < box_size - 1.0f)
		{
			sphere_center[2] += 0.05f;
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
	case 'R':
		wind[0] = 0.0f;
		wind[1] = 0.0f;
		wind[2] = 0.0f;
		break;
	case 'm':
	case 'M':
		mode = !mode;
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
