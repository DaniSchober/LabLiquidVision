#include <sstream>
#include <random>
#include <chrono>
#include "../mesh_query/mesh_query.h"
#include <windows.h>
#include <stdio.h>

// This scene is used to calibrate the particle to volume ratio for the pouring simulations. The calibration container has a volume of 500 mL.

class Calibration : public Scene
{
public:
	
	std::vector<bool> particle_never_left;

	string container_path;
	string output_path;
	ofstream theta_vs_volume_file;
	ofstream theta_vs_slosh_time_file;
	ofstream TCP_file;

	NvFlexTriangleMeshId mesh_receiver, container_mesh;
	Vec3 receive_pos;
	Quat receive_rot;
	float mTime;
	float startTime;
	int frame_count = 0;
	int num_particles = -1;
	float prev_theta = 0;
	float rotationSpeed;
	float radius_CoR;
	float TCP_x;
	float TCP_y;
	float pos_x;
	float pos_y;
	float prev_pos_x;
	float prev_pos_y;
	float emitterSize;
	int row = 1;
	float pause_time = 1.0;
	bool pause_complete = false;
	float stop_angle;
	float next_stop_threshold = 0;
	bool stopped = false;
	bool return_activated = false;
	int prev_particle_count = -1;
	float pause_start_time = 0;
	float pause_start = 0;

	Calibration(const char* name, string object_path) : 
		Scene(name), 
		container_path(object_path) // path to the pouring container (.obj type)
		{}

	virtual void Initialize()
	{
		printf("Init\n");

		// get data from config_file of pouring container if it exists --> maybe save there the TCP and center of rotation positions
		ifstream config_file(container_path + ".cfg");
		if (config_file.is_open()) {
			printf("Config file found\n");
		}
		printf("Path %s", container_path.c_str());
		config_file >> TCP_x;
		config_file >> TCP_y;
		config_file >> radius_CoR;
		config_file >> emitterSize;

		pos_x = TCP_x;
		pos_y = TCP_y;
		prev_pos_x = TCP_x;
		prev_pos_y = TCP_y;

		printf("TCP %f %f, radius: %f, emitter_size: %f\n", TCP_x, TCP_y, radius_CoR, emitterSize);

		// create generator for random rotation speed between sqrt(0.005) and sqrt(0.1)
		std::default_random_engine generator(chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<float> distribution(sqrt(0.005), sqrt(0.1));
		rotationSpeed = distribution(generator);
		rotationSpeed *= rotationSpeed;

		// create generator for random stop angle betweem 45 and 135 degrees
		std::default_random_engine generator2(chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<float> distribution2(10.0, 60.0);
		stop_angle = distribution2(generator2);
		//stop_angle = 50;
		//stop_angle = 180;

		ofstream param_file;
		param_file.open(output_path + "_params.txt");
		param_file << "rotation_speed " << rotationSpeed << std::endl;
		param_file << "stop_angle " << stop_angle << std::endl;
		param_file.close();
		printf("Stop angle %f rot speed %f", stop_angle, rotationSpeed);


		TCP_file.open(output_path + "_TCP.txt");
		TCP_file << "pos_x, " << "pos_y, " << "theta (rad)" << std::endl;

		// set drawing options
		g_drawPoints = false;
		g_drawEllipsoids = true; // Draw as fluid
		g_wireframe = false;
		g_drawDensity = false;
		g_drawSprings = false;
		g_lightDistance = 5.0f;
		mTime = 0.0f;


		//////////////////////////////////////////////////////// Add container ////////////////////////////////////////////////////////////////////////////////
		// Import mesh of pouring container 
		Mesh* container = ImportMesh(GetFilePathByPlatform((container_path+".obj").c_str()).c_str());

		// rotate the coordinate system of the pouring container around 90 degrees (so it's horizontal)
		float angle = 0.0f; 
		// Define the axis of rotation
		Vec3 axis = Vec3(0.0f, 0.0f, 1.0f);
		// Define the rotation quaternion
		Quat rot = QuatFromAxisAngle(axis, angle);
		// Rotate the mesh
		container->Transform(RotationMatrix(rot));

		// create triangle mesh of pouring container
		container_mesh = CreateTriangleMesh(container);

		// set the initial pouring container position and orientation (TCP_y is the height of the TCP of the container compared to the ground plane)
		Vec3 pos = Vec3(TCP_x, TCP_y, 0.0f);
		Quat rot1 = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 1.0f - cosf(0));

		// add initial triangle mesh to the scene
		AddTriangleMesh(container_mesh, pos, rot1, 1.0f);
		
		// delete receiver and pourer object 
		delete container;

		//////////////////////////////////////////////////// Set fluid parameters and create emitter ///////////////////////////////////////////////////////////////////
		float radius = 0.1f; // radius of particles
		float restDistance = radius*0.6f;
		Vec3 lower = (0.0f, 10.0f, 0.0f);
		
		g_numSubsteps = 10;
		g_fluidColor = Vec4(0.2f, 0.6f, 0.9f, 1.0f); // blue
		//g_fluidColor = Vec4(0.9f, 0.5f, 0.5f, 0.5f); // red
		g_params.radius = radius;
		g_params.dynamicFriction = 0.0f;
		g_params.dissipation = 0.0f;
		g_params.numPlanes = 1;
		g_params.restitution = 0.001f;
		g_params.fluidRestDistance = restDistance;
		g_params.numIterations = 3;
		g_params.anisotropyScale = 30.0f;
		g_params.fluid = true;
		g_params.relaxationFactor = 1.0f;
		g_params.smoothing = 0.5f;
		g_params.collisionDistance = g_params.radius*.25f;
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.05f;
	
		// parameters from paper PourNet
		g_params.viscosity = 0.01f;
		g_params.cohesion = 0.001f;
		g_params.vorticityConfinement = 80.0f;
		g_params.surfaceTension = 0.005f;
		g_params.adhesion = 0.0001f;
		// parameters from paper PourNet end


		////////////////////// Change location of water emitter //////////////////////////////////////////////
		Vec3 center = Vec3(TCP_x, TCP_y+4.0157, 0.0f); // puts emitter in the middle of the pouring container
		Emitter e; // create emitter object
		e.mEnabled = true;
		e.mWidth = int(emitterSize / restDistance);
		e.mPos = Vec3(center.x, center.y, center.z);
		e.mDir = Vec3(0.0f, -1.0f, 0.0f); // sets emitting direction to downwards in y direction
		e.mRight = Vec3(-1.0f, 0.0f, 0.0f);
		e.mSpeed = 0.05f;
		g_sceneUpper.z = 5.0f;
		g_emitters.push_back(e);

		g_numExtraParticles = 200000; //(int)(75 * 3.14 * area * area / radius / radius / radius) + 2000;
		printf("Num particles %d \n", g_numExtraParticles);
		// The particles are spawned once every eight of a second.  It creates a number of
		// particles proportional to the area of the emitter.  Five seconds is then added to
		// let the water settle
		//startTime = 1.0f * g_numExtraParticles / e.mWidth / e.mWidth / 8 + 20;// +5;
		startTime = g_numExtraParticles / e.mWidth / e.mWidth / 8 + 5;
		g_emit = false;

		theta_vs_volume_file.open(output_path +".text");
		theta_vs_volume_file << "inside_count" << "\t" << "num_particles" << "\t" << "theta (rad)" << "\t" << "time (s)" << "\n";
		theta_vs_slosh_time_file.open(output_path + "_wait_times.text");
		theta_vs_slosh_time_file << "wait_time" << "\t" << "theta" << "\n";


		for (int i = 0; i < g_numExtraParticles; i++) {
			particle_never_left.push_back(true);
		}




		printf("Initialized \n");
	}


	void Update()
	{
		// Defaults to 60Hz
		mTime += g_dt;

		// the scene settle before moving
		if (mTime > 0.5) {
			g_emit = true;
		}
		else {
			g_emit = false;
			return;
		}
		
		
		stopped = false;

		//AddTriangleMesh(container_mesh, pos, rot, 1.0f);
		UpdateShapes();
	}
};
