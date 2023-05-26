#include <sstream>
#include <random>
#include <chrono>
#include "../mesh_query/mesh_query.h"
#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <fstream>


class Pouring_Flask : public Scene
{
public:
	
	std::vector<bool> particle_never_left;

	string pouring_container_path;
	string output_path;
	ofstream theta_vs_volume_file;
	ofstream theta_vs_slosh_time_file;
	ofstream TCP_file;
	ofstream summary_file;

	NvFlexTriangleMeshId mesh_receiver, pouring_mesh;
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
	float numStartParticles;
	float start_volume = 0;
	int row = 1;

	float pause_time = 0.0;
	bool pause_complete = false;
	float stop_angle;
	float next_stop_threshold = 0;
	bool stopped = false;
	bool return_activated = false;
	int prev_particle_count = -1;
	float pause_start_time = 0;
	float pause_start = 0;

	Pouring_Flask(const char* name, string object_path, string out_path, int start_vol, float stop_duration, float stop_angle): 
		Scene(name), 
		pouring_container_path(object_path), 	// path to the pouring container (.obj type)
		output_path(out_path),    				// path to the output folder and file name
		start_volume(start_vol), 				// volume of liquid in mL at the beginning of the simulation
		pause_time(stop_duration), 				// time in seconds to pause the simulation after the liquid has reached the stop angle
		stop_angle(stop_angle)					// stop angle in degrees
		{}

	virtual void Initialize()
	{
		printf("Init\n");

		// get data from config_file of pouring container if it exists --> maybe save there the TCP and center of rotation positions
		ifstream config_file(pouring_container_path + ".cfg");
		if (config_file.is_open()) {
			printf("Config file found\n");
			// read in the data from the config file

		}
		printf("Path %s \n", pouring_container_path.c_str());
		//float TCP_x, TCP_y, radius;
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
		rotationSpeed = 0.03;
		//rotationSpeed = 0.05;

		// create generator for random stop angle betweem 45 and 135 degrees
		std::default_random_engine generator2(chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<float> distribution2(10.0, 60.0);
		//stop_angle = distribution2(generator2);
		//stop_angle = 5;
		//stop_angle = 180;

		ofstream param_file;
		param_file.open(output_path + "_params.txt");
		param_file << "rotation_speed " << rotationSpeed << std::endl;
		param_file << "stop_angle " << stop_angle << std::endl;
		param_file.close();
		printf("Stop angle %f rot speed %f \n", stop_angle, rotationSpeed);


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

		//////////////////////////////////////////////////////// Add receiving container /////////////////////////////////////////////////////////////////////////////
		
		Mesh* receiver = ImportMesh(GetFilePathByPlatform("../../data/Assembly_Receiver.obj").c_str());
		mesh_receiver = CreateTriangleMesh(receiver);

		receive_pos = Vec3(0.0f, 0.1f, 0.0f); // x, y, z (y is up)! Changing position of the receiving container
		//receive_rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -0.244f); // turn around z-axis around 14 degrees (in radians 0.244f)
		receive_rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 0.0f); // turn around z-axis around 14 degrees (in radians 0.244f)
		AddTriangleMesh(mesh_receiver, receive_pos, receive_rot, 1.0f); // change scale of the receiving container

		//////////////////////////////////////////////////////// Add pouring container ////////////////////////////////////////////////////////////////////////////////
		// Import mesh of pouring container 
		Mesh* pourer = ImportMesh(GetFilePathByPlatform((pouring_container_path+".obj").c_str()).c_str());

		// rotate the coordinate system of the pouring container around 90 degrees (so it's horizontal)
		float angle = -1.5708f; 
		// Define the axis of rotation
		Vec3 axis = Vec3(0.0f, 0.0f, 1.0f);
		// Define the rotation quaternion
		Quat rot = QuatFromAxisAngle(axis, angle);
		// Rotate the mesh
		pourer->Transform(RotationMatrix(rot));

		// create triangle mesh of pouring container
		pouring_mesh = CreateTriangleMesh(pourer);

		// set the initial pouring container position and orientation (TCP_y is the height of the TCP of the container compared to the ground plane)
		Vec3 pos = Vec3(TCP_x, TCP_y, 0.0f);
		Quat rot1 = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 1.0f - cosf(0));

		// add initial triangle mesh to the scene
		AddTriangleMesh(pouring_mesh, pos, rot1, 1.0f);
		
		// delete receiver and pourer object 
		delete receiver;
		delete pourer;

		//////////////////////////////////////////////////// Set fluid parameters and create emitter ///////////////////////////////////////////////////////////////////
		float radius = 0.1f; // radius of particles
		float restDistance = radius*0.6f;
		Vec3 lower = (0.0f, 10.0f, 0.0f);
		//int x_count = (int)(1.0f / restDistance); //not sure if needed
		//int y_count = (int)(1.0f / restDistance);
		//int z_count = (int)(1.0f / restDistance);
		//int water_phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
		
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
		Vec3 center = Vec3(TCP_x-0.7, TCP_y + 0.5 , 0.0f); // puts emitter in the middle of the pouring container
		Emitter e; // create emitter object
		e.mEnabled = true;
		e.mWidth = int(emitterSize / restDistance);
		e.mPos = Vec3(center.x, center.y, center.z);
		e.mDir = Vec3(0.0f, -1.0f, 0.0f); // sets emitting direction to downwards in y direction
		e.mRight = Vec3(-1.0f, 0.0f, 0.0f);
		e.mSpeed = 0.05f;
		g_sceneUpper.z = 5.0f;
		g_emitters.push_back(e);

		g_numExtraParticles = start_volume*400; // number of particles in the emitter
		numStartParticles = g_numExtraParticles;
		printf("Num particles %d \n", g_numExtraParticles);

		// The particles are spawned once every eight of a second. 10 seconds is added to let the water settle
		startTime = g_numExtraParticles / 1500 + 10; // time to emit particles and let the liquid settle
		g_emit = false;

		theta_vs_volume_file.open(output_path +".text");
		theta_vs_volume_file << "inside_count" << "\t" << "num_particles" << "\t" << "theta (rad)" << "\t" << "time (s)" << "\n";

		for (int i = 0; i < g_numExtraParticles; i++) {
			particle_never_left.push_back(true);
		}

		printf("Initialized \n");
	}



	bool InPouringContainer(Vec4 position, float theta) {
		return position.y > TCP_y - 1.5; // only checks if the particle is above a certain height
	}

	bool InReceivingFlask(Vec4 position){
		return position.y > 0.22188 && position.y < 7;
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
		
		if (num_particles == -1 && mTime > startTime - 0.5) {
			int not_poured_count = 0;
			for (int i = 0; i < g_buffers->positions.count; i++) {
				if (InPouringContainer(g_buffers->positions[i], 0)) {
					not_poured_count++;
				}
				else {
					g_buffers->positions[i].x += 1000; // Clears out the overflow before simulation starts
					
				}
			}
			num_particles = not_poured_count;
		}

		g_buffers->shapeGeometry.resize(1);
		g_buffers->shapePositions.resize(1);
		g_buffers->shapeRotations.resize(1);
		g_buffers->shapePrevPositions.resize(1);
		g_buffers->shapePrevRotations.resize(1);
		g_buffers->shapeFlags.resize(1);

		float time = Max(0.0f, mTime - startTime);
		float lastTime = Max(0.0f, time - g_dt);
		float endTime = 3.14 / rotationSpeed;
		float theta = 0;

		// If true, the cup will stop every stop_angle degrees and wait for the water level to stop changing
		bool continous = false;
		float stop_distance = stop_angle*3.14/180;
		float real_prev_theta = prev_theta;

		///////////////////////////////////////////////////////////////////// Movements ///////////////////////////////////////////////////////////////////////////

		if (continous) {
			theta = 3.14f * (1.0f - cosf(rotationSpeed*time)); 
			time = Min(time, endTime);
			lastTime = Min(lastTime, endTime);
		}
		else if (time > 0 && !return_activated) {
			theta = prev_theta;
			prev_pos_x = pos_x;
			prev_pos_y = pos_y;

			if (!stopped) {

				theta = prev_theta + rotationSpeed * g_dt;
				pos_x = TCP_x + radius_CoR * (1 - cos(theta));
				pos_y = TCP_y + radius_CoR * sin(theta);
				
				printf("pos_x: %f pos_y: %f theta: %f\n", pos_x, pos_y, theta*57.2957795);

				if (theta > next_stop_threshold) {
					printf("Stopped activated: theta: %f\n", theta);
					pause_start_time = time;
					next_stop_threshold += stop_distance;
					stopped = true;
					prev_particle_count = -1;
				}
			}
			
			prev_theta = theta;
		}
		else if (return_activated && prev_theta > 0) {
			if (!pause_complete) {
				printf("Pausing at max angle: %f for %.2f time\n", prev_theta, pause_time);
				if (time - pause_start > pause_time) {
					pause_complete = true;
					theta = prev_theta;
					prev_pos_x = pos_x;
					prev_pos_y = pos_y;
				}
				else {
					theta = prev_theta;
					pos_x = prev_pos_x;
					pos_y = prev_pos_y;
				}
			}
			else {
				theta = prev_theta;
				prev_pos_x = pos_x;
				prev_pos_y = pos_y;

				if (!stopped) {

					theta = prev_theta - rotationSpeed * g_dt;
					pos_x = TCP_x + radius_CoR * (1 - cos(theta));
					pos_y = TCP_y + radius_CoR * sin(theta);

					printf("pos_x: %f pos_y: %f theta: %f\n", pos_x, pos_y, theta*57.2957795);

					if (theta < next_stop_threshold) {
						printf("Stopped activated: theta: %f\n", theta);
						pause_start_time = time;
						next_stop_threshold -= stop_distance;
						stopped = true;
						prev_particle_count = -1;
					}
				}

				prev_theta = theta;
			}			
		}

		else if (return_activated && prev_theta <= 0) {
			printf("Return finished\n");

			// Write the data to file
			// open output.csv file
			ofstream summary_file;
			summary_file.open("../../output/summary.csv");
			// add new line
			summary_file << "This is the first cell in the first column.\n";
			summary_file << "a,b,c,\n";
			printf("Writing to file\n");
			// close the output.csv file
			summary_file.close();
			g_scene_finished = true;
			theta_vs_volume_file.close();
			theta_vs_slosh_time_file.close();
			return;
		}	

		/////////////////////////////////////////////////////////////////////////////////// End Movements /////////////////////////////////////////////////////////////////////////////////////////////////
				
		// calculate the amount of particles still in the container
		int not_poured_count = 0;
		int received_count = 0;

		for (int i = 0; i < g_buffers->positions.count; i++) {
			if (particle_never_left[i] && InPouringContainer(g_buffers->positions[i], theta)) {
				not_poured_count++;
			}
			else if (InReceivingFlask(g_buffers->positions[i])) {
				received_count++;
			}	
			else if (time > 0.0) {  // Only flag particles once everything has been inited
				particle_never_left[i] = false;
				numStartParticles--;
			}
			
		}

		// write results for each step in file
		if (mTime > startTime) {
			theta_vs_volume_file << not_poured_count << "\t\t" << num_particles << "\t\t" << theta << "\t" << time << "\n";
			TCP_file << pos_x-TCP_x << ", " << pos_y-TCP_y << ", " << theta << "\n";
		}
		frame_count++;
		
		// if stop angle is reached, start pause time and activate the return
		if (theta > (stop_angle-0.05) * 3.14 / 180 && !return_activated) {
					printf("Activated return\n");
					pause_start = time;
					return_activated = true;
				}
		stopped = false;

		// Open the output file for writing:
    	FILE *file = fopen("../../output/output.csv", "w");
		if (row == 0){
			fprintf(file, "pos_x,pos_y,theta\n");
		}
		else{
			fprintf(file, "%f,%f,%f\n", pos_x-TCP_x, pos_y-TCP_y, theta);
		}
		fclose(file);

		
		// write pouring results in _params file
		ofstream param_file;
		param_file.open(output_path + "_params.txt");
		param_file << "rotation_speed " << rotationSpeed << std::endl;
		param_file << "stop_angle " << stop_angle << std::endl;
		param_file << "pause_time " << pause_time << std::endl;
		param_file << "num_particles " << g_numExtraParticles << std::endl;
		param_file << "start_particles " << num_particles << std::endl;
		param_file << "poured_particles " << num_particles - not_poured_count << std::endl;
		param_file << "received_particles " << received_count << std::endl;
		param_file << "spilled_particles " << num_particles - not_poured_count - received_count << std::endl;
		param_file << "not_poured_particles " << not_poured_count << std::endl;
		param_file << "\n" << std::endl;
		param_file << "start_volume " << (num_particles/400.0) << " mL" << std::endl;
		param_file << "poured_volume " << (num_particles - not_poured_count)/400.0 << " mL" << std::endl;
		param_file << "received_volume " << received_count/400.0 << " mL" << std::endl;
		param_file << "spilled_volume " << (num_particles - not_poured_count - received_count)/400.0 << " mL" << std::endl;
		param_file << "not_poured_volume " << not_poured_count/400.0 << " mL" << std::endl;
		param_file.close();

		// update positions of pouring container
		Vec3 prevPos = Vec3(prev_pos_x, prev_pos_y, 0.0f);
		Vec3 pos = Vec3(pos_x, pos_y, 0.0f);
		Quat rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -theta); 
		Quat prevRot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -real_prev_theta);
		AddTriangleMesh(pouring_mesh, pos, rot, 1.0f);
		g_buffers->shapePrevPositions[1] = Vec4(prevPos, 0.0f);
		g_buffers->shapePrevRotations[1] = prevRot;	
		UpdateShapes();
	}
};
