#include <sstream>
#include <random>
#include <chrono>
#include "../mesh_query/mesh_query.h"
#include <windows.h>

class Pouring : public Scene
{
public:

	std::vector<double> bounding_mesh_vertices;
	std::vector<int> bounding_mesh_indices;

	std::vector<double> bounding_mesh_vertices_transformed;
	const MeshObject* bounding_mesh;
	
	std::vector<bool> particle_never_left;

	string pouring_container_path;
	string output_path;
	float unscaled_glass_width, glass_width, glass_height, glass_top_elevation;
	ofstream theta_vs_volume_file;

	ofstream theta_vs_slosh_time_file;

	NvFlexTriangleMeshId mesh_receiver, pouring_mesh;
	Vec3 receive_pos;
	Quat receive_rot;
	float mTime;
	float startTime;
	int frame_count = 0;
	int num_particles = -1;
	float prev_theta = 0;
	float rotationSpeed;
	float x_trans = -5.1;
	float x_translation = 0.0;
	float y_translation = 0.0;

	//float y_trans = 0.0;

	float stop_angle;
	float next_stop_threshold = 0;
	bool stopped = false;
	int prev_particle_count = -1;
	float pause_start_time = 0;
	int pause_no_change_count = 0;

	float vis = -1;
	float coh = -1;

	Pouring(const char* name, string object_path, string out_path, float viscosity, float cohesion) : 
		Scene(name), 
		pouring_container_path(object_path), 	// path to the pouring container (.obj type)
		output_path(out_path),    				// path to the output folder
		glass_top_elevation(8.7), 				// how much the pouring container top is above the receiving container
		vis(viscosity), coh(cohesion)			//, rotationSpeed(speed)
		{}


	void LoadBoundingMesh(std::string mesh_path)
	{
		ifstream mesh(mesh_path);

		std::string line;
		while (std::getline(mesh, line)) {
			std::istringstream iss(line);
			char start;
			iss >> start;

			if (start == 'v') { // vertices
				double coord;
				iss >> coord;
				bounding_mesh_vertices.push_back(coord);
				iss >> coord;
				bounding_mesh_vertices.push_back(coord);
				iss >> coord;
				bounding_mesh_vertices.push_back(coord);
			}
			else if (start == 'f') { // Faces
				int index;
				iss >> index;
				bounding_mesh_indices.push_back(index);
				iss >> index;
				bounding_mesh_indices.push_back(index);
				iss >> index;
				bounding_mesh_indices.push_back(index);
			}
		}
	}

	virtual void Initialize()
	{
		printf("Init\n");

		// create generator for random rotation speed between sqrt(0.005) and sqrt(0.1)
		std::default_random_engine generator(chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<float> distribution(sqrt(0.005), sqrt(0.1));
		rotationSpeed = distribution(generator);
		rotationSpeed *= rotationSpeed;
		rotationSpeed = 0.05;

		// create generator for random stop angle betweem 45 and 135 degrees
		std::default_random_engine generator2(chrono::steady_clock::now().time_since_epoch().count());
		std::uniform_real_distribution<float> distribution2(45.0, 135.0);
		stop_angle = distribution2(generator2);
		stop_angle = 40;
		//stop_angle = 180;

		ofstream param_file;
		param_file.open(output_path + "_params.txt");
		param_file << "rotation_speed " << rotationSpeed << std::endl;
		param_file << "stop_angle " << stop_angle << std::endl;
		param_file.close();
		printf("Stop angle %f rot speed %f", stop_angle, rotationSpeed);

		// set drawing options
		g_drawPoints = false;
		g_drawEllipsoids = true; // Draw as fluid
		g_wireframe = false;
		g_drawDensity = false;
		g_drawSprings = false;
		g_lightDistance = 5.0f;
		mTime = 0.0f;
		

		//////////////////////////////////////////////////////// Add receiving container /////////////////////////////////////////////////////////////////////////////
		
		// Import mesh of the recieving container
		//Mesh* bowl = ImportMesh(GetFilePathByPlatform("../../data/bowl.obj").c_str());

		// change bowl size
		//float bowl_width = 2.5;

		//bowl->Normalize(bowl_width);
		//bowl->CalculateNormals();
		//bowl->Transform(TranslationMatrix(Point3(-bowl_width / 2, 0.0f, -bowl_width/2)));

		// create triangle mesh of receiving container
		//mesh = CreateTriangleMesh(bowl);

		// set position and rotation of receiving container and add it to scene
		//recieve_pos = Vec3(0.0f, 0.0f, 0.0f);
		//recieve_rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 0.0f);
		//AddTriangleMesh(mesh, recieve_pos, recieve_rot, 1.0f);
		
		Mesh* receiver = ImportMesh(GetFilePathByPlatform("../../data/CellFlask.obj").c_str());
		mesh_receiver = CreateTriangleMesh(receiver);

		receive_pos = Vec3(0.0f, 0.1f, 0.0f); // x, y, z (y is up)! Changing position of the receiving container
		receive_rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -0.244f); // turn around z-axis around 14 degrees (in radians 0.244f)
		AddTriangleMesh(mesh_receiver, receive_pos, receive_rot, 1.0f); // change scale of the receiving container

		//////////////////////////////////////////////////////// Add pouring container ////////////////////////////////////////////////////////////////////////////////
		// get data from config_file of pouring container if it exists
		ifstream config_file(pouring_container_path + ".cfg");
		double min_radius, max_radius, height, area;
		//config_file >> min_radius;
		//config_file >> max_radius;
		//config_file >> height;
		//config_file >> area;
		min_radius = 0.145;	// minimum radius of pouring container
		max_radius = 0.178;	// maximum radius pf pouring container
		height = 0.1; 		// height of the pouring container
		area = 0.066; 		// area of the pouring container
		printf("\nMin_radius %f, Max_radius %f, Height %f\n", min_radius, max_radius, height);
		
		// set glass width to max_radius of the object
		glass_width = max_radius;

		// set glass height to height of the object
		glass_height = height;

		// Import mesh of pouring container 
		Mesh* glass = ImportMesh(GetFilePathByPlatform((pouring_container_path+".obj").c_str()).c_str());

		// move the pouring container downwards, so it can rotate around another axis: basically moving the reference coordinate system of the object (idea: half the height of the object downwards, I think it's in inches)
		glass->Transform(TranslationMatrix(Point3( 0.0, -3.93, 0.0)));
		//glass->CalculateNormals();
		// Define the angle of rotation in radians
		float angle = -1.5708f; // for example, rotate 1 radian

		// Define the axis of rotation
		Vec3 axis = Vec3(0.0f, 0.0f, 1.0f);

		// Define the rotation quaternion
		Quat rot = QuatFromAxisAngle(axis, angle);

		// Rotate the mesh
		glass->Transform(RotationMatrix(rot));


		pouring_mesh = CreateTriangleMesh(glass);

		// set pouring container position and orientation (glass_top_elevation is the height of the top of the container compared to the ground plane)
		Vec3 pos = Vec3(x_trans, glass_top_elevation, 0.0f);
		//Vec3 pos = Vec3(50.0f, 30.0f, 50.0f);
		//Vec3 pos = Vec3(0.0f, glass_height, 0.0f);
		Quat rot1 = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 1.0f - cosf(0));
		//Quat rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), 2.0f);
		AddTriangleMesh(pouring_mesh, pos, rot1, 1.0f);
		

		// delete bowl and glass object 
		delete receiver;
		delete glass;

		//////////////////////////////////////////////////// Set fluid parameters and create emitter ///////////////////////////////////////////////////////////////////
		float radius = 0.1f;
		float restDistance = radius*0.6f;
		Vec3 lower = (0.0f, 10.0f, 0.0f);
		int x_count = (int)(1.0f / restDistance);
		int y_count = (int)(1.0f / restDistance);
		int z_count = (int)(1.0f / restDistance);
		int water_phase = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
		
		g_numSubsteps = 10; //2;
		g_params.radius = radius;
		g_params.dynamicFriction = 0.0f; //0.1f;
		g_params.dissipation = 0.0f;
		g_params.numPlanes = 1;
		g_params.restitution = 0.001f; // new added
		g_params.fluidRestDistance = restDistance;
		//g_params.viscosity = vis;// 0.10f; // 0.0f
		//g_params.cohesion = coh;// 0.0f;// 0.02f;
		g_params.numIterations = 3;
		g_params.anisotropyScale = 30.0f;
		g_params.fluid = true;
		g_params.relaxationFactor = 1.0f;
		g_params.smoothing = 0.5f;
		g_params.collisionDistance = g_params.radius*.25f; //g_params.radius*0.5f;
		g_params.shapeCollisionMargin = g_params.collisionDistance*0.05f;
	
		// parameters from paper PourNet
		g_params.viscosity = 0.01f; // changed
		g_params.cohesion = 0.001f; // changed
		g_params.vorticityConfinement = 80.0f; // new
		g_params.surfaceTension = 0.005f; // new
		g_params.adhesion = 0.0001f; // new
		// parameters from paper PourNet end

		float emitterSize = 3.5f; //min_radius* 2 + 0.01; // 0.5f;
		printf("Emitter size  %f %f %f\n", emitterSize, min_radius, radius);

		//////////////////////////////////////////////////////////////////////////////////////////////////////
		////////////////////// Change location of water emitter //////////////////////////////////////////////

		//Vec3 center = Vec3(0.0, glass_top_elevation - glass_height/2.0, 0.0); // Emits particles in the center of the object
		Vec3 center = Vec3(x_trans-0.7, glass_top_elevation + 1.0f, -2.35f);

		Emitter e;
		e.mEnabled = true;
		e.mWidth = int(emitterSize / restDistance);
		e.mPos = Vec3(center.x, center.y, center.z);
		e.mDir = Vec3(0.0f, -1.0f, 0.0f);
		e.mRight = Vec3(-1.0f, 0.0f, 0.0f);
		e.mSpeed = 0.05f; //restDistance*0.2f / g_dt;// (restDistance*2.f / g_dt); //0.06*0.2*8 = 0,096

		g_sceneUpper.z = 5.0f;

		g_emitters.push_back(e);

		g_numExtraParticles = 20000; //(int)(75 * 3.14 * area * area / radius / radius / radius) + 2000;
		// 153000 particles = 322,94896 mm3ffff
		//g_numExtraParticles = (int)(75 * 3.14 * area * area / radius / radius / radius) + 2000;
		printf("Num particles %d \n", g_numExtraParticles);
		// The particles are spawned once every eight of a second.  It creates a number of
		// particles proportional to the area of the emitter.  Five seconds is then added to
		// let the water settle
		//startTime = 1.0f * g_numExtraParticles / e.mWidth / e.mWidth / 8 + 20;// +5;
		startTime = 15.0f;
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



	bool InBoundingBox(Vec4 position, float theta) {
		return position.y > glass_top_elevation - 1.5;
		//return position.y > glass_top_elevation - glass_height
		//	&& (position.x >  -cosf(theta) * glass_width/2);
		//return position.x > tanf(1.57 + theta) * position.y - glass_width / 2 / cosf(theta);

		float x1 = -glass_width / 2 * cosf(theta); // Upper left
		float y1 = glass_top_elevation - glass_width/2 *sinf(theta);
		float x2 = x1 + glass_height * cosf(1.57 - theta);// Lower left
		float y2 = y1 - glass_height * sinf(1.57 - theta);

        float x3 = glass_width / 2 * cosf(theta); // Upper right
        float y3 = glass_top_elevation + glass_width/2 *sinf(theta);
        float x4 = x3 + glass_height * cosf(1.57 - theta); // lower right
        float y4 = y3 - glass_height * sinf(1.57 - theta);

		//printf("%f %f %f %f\n", x1, y1, x2, y2);
		//return position.x * (y2 - y1) - position.y *(x2 - x1) < x1*y2 - x2*y1;
		return (x2 - x1)*(position.y - y1) > (y2 - y1)*(position.x - x1) && // left bound
			(x4 - x3)*(position.y - y3) < (y4 - y3)*(position.x - x3) && // right bound
			(x3 - x1)*(position.y - y1) < (y3 - y1)*(position.x - x1) && // top bound
			(x4 - x2)*(position.y - y2) > (y4 - y2)*(position.x - x2); // bottom bound
	}

	void Update()
	{
		// Defaults to 60Hz
		mTime += g_dt;
		// print g_dt
		printf("mTime %f \n", mTime);
		// the scene settle before moving
		if (mTime > 0.5) {
			g_emit = true;
		}
		else {
			g_emit = false;
			return;
		}
		
		if (num_particles == -1 && mTime > startTime - 0.5) {
			int inside_count = 0;
			for (int i = 0; i < g_buffers->positions.count; i++) {
				if (InBoundingBox(g_buffers->positions[i], 0)) {
					inside_count++;
				}
				else {
					g_buffers->positions[i].x += 1000;// Clears out the overflow before simulation starts
				}
			}
			num_particles = inside_count;
		}

		//ClearShapes();

		g_buffers->shapeGeometry.resize(1);
		g_buffers->shapePositions.resize(1);
		g_buffers->shapeRotations.resize(1);
		g_buffers->shapePrevPositions.resize(1);
		g_buffers->shapePrevRotations.resize(1);
		g_buffers->shapeFlags.resize(1);

		//int num_particles = g_buffers->positions.count;
		//printf("num_particles: %d\n", num_particles);
		//printf("Particle0 %f %f %f\n", g_buffers->positions[0].x, g_buffers->positions[0].y, g_buffers->positions[0].z);
		
		//startTime = 50.0f;


		float time = Max(0.0f, mTime - startTime);
		float lastTime = Max(0.0f, time - g_dt);


		const float translationSpeed = 0.1f;// 1.0f;

		float endTime = 3.14 / rotationSpeed;

		

		
		float theta = 0;
		// If true, the cup will stop every stop_angle degrees and wait for the water level to stop changing
		bool continous = false;

		
		float stop_distance = stop_angle*3.14/180;
		float max_theta_speed = rotationSpeed;// 0.0005; //0.0003
		float fraction_at_max_speed = 0.0;// 0.95;
		float real_prev_theta = prev_theta;
		if (continous) {
			theta = 3.14f * (1.0f - cosf(rotationSpeed*time)); // rotationSpeed*time;
			time = Min(time, endTime);
			lastTime = Min(lastTime, endTime);
		}
		else if (time > 0) {
			theta = prev_theta;
			if (!stopped) {
				theta = prev_theta + rotationSpeed * g_dt;

				if (theta > next_stop_threshold) {
					pause_start_time = time;
					next_stop_threshold += stop_distance;
					stopped = true;
					prev_particle_count = -1;
				}
			}
			
			prev_theta = theta;
		}

		if (theta > 3.) {
			g_scene_finished = true;
			theta_vs_volume_file.close();
			theta_vs_slosh_time_file.close();
			return;
		}
		
		ofstream frame_particle_locations, frame_container_orientation;
		bool output_render_data = false;
		if (frame_count % 3 == 0 && output_render_data) {
			frame_particle_locations.open(output_path + "_particles_" + std::to_string(frame_count) + ".obj");
			frame_container_orientation.open(output_path + "_container_" + std::to_string(frame_count) + ".obj");
			double theta_degree = theta / M_PI * 180;
			frame_container_orientation << "v 0 0 " << theta_degree << std::endl;
			frame_container_orientation << "v 0 " << glass_top_elevation << " 0" << std::endl;
		}
		
		//const MeshObject* bounding_mesh = UpdateBoundingMesh(theta);
		//UpdateBoundingMesh(theta);
		int inside_count = 0;
		for (int i = 0; i < g_buffers->positions.count; i++) {
			if (particle_never_left[i] && InBoundingBox(g_buffers->positions[i], theta)) {
				inside_count++;
				//printf("g %f %f %f\n", g_buffers->positions[i].x, g_buffers->positions[i].y, g_buffers->positions[i].z);
			}
			else if (time > 0.0) {  // Only flag particles once everything has been inited
				particle_never_left[i] = false;
			}
			//if (InBoundingMesh(g_buffers->positions[i], bounding_mesh)) {
			//	inside_count_2++;
			//}
			//else {
			//	if (InBoundingBox(g_buffers->positions[i], theta)) {
			//		//printf("%f %f %f\n", g_buffers->positions[i].x, g_buffers->positions[i].y, g_buffers->positions[i].z);
			//	}
			//	g_buffers->positions[i].x = 1000;
			//}

			if (frame_count % 3 == 0 && output_render_data) {
				frame_particle_locations << "v " << g_buffers->positions[i].x << " "
					<< g_buffers->positions[i].y << " "
					<< g_buffers->positions[i].z << std::endl;
			}
		}
		if (frame_count % 3 == 0 && output_render_data) {
			frame_particle_locations.close();
			frame_container_orientation.close();
		}
		//printf("%f %f %f\n", theta, tanf(theta), glass_width / 2 / cosf(theta));

		if (mTime > startTime) {
			theta_vs_volume_file << inside_count << "\t\t" << num_particles << "\t\t" << theta << "\t" << time << "\n";
		}
		frame_count++;
		//printf("%f %d / %d / %d particles in container at %f %d\n", mTime,inside_count, num_particles, g_buffers->positions.count, theta, frame_count);
		
		if (!continous && stopped) {
			//printf("Paused\n");
			if (prev_particle_count == inside_count) {
				pause_no_change_count++;
			}
			else
			{
				pause_no_change_count = 0;
			}

			if(pause_no_change_count > 20) {
				printf("Unpaused %f %f\n", (time - 100*g_dt - pause_start_time), theta);
				pause_no_change_count = 0;
				stopped = false;
				theta_vs_slosh_time_file << (time - 100 * g_dt - pause_start_time) << " " << theta << "\n";

				if (theta > (stop_angle - 2) * 3.14 / 180) {
					//double milliseconds = 100;
					//Sleep(milliseconds);
					g_scene_finished = true;
					theta_vs_volume_file.close();
					theta_vs_slosh_time_file.close();
					return;
				}
				
			}
			prev_particle_count = inside_count;
		}
		
		ofstream param_file;
		param_file.open(output_path + "_params.txt");
		param_file << "rotation_speed " << rotationSpeed << std::endl;
		param_file << "stop_angle " << stop_angle << std::endl;
		param_file << "num_particles " << g_numExtraParticles << std::endl;
		param_file << "poured_particles " << g_numExtraParticles - inside_count << std::endl;
		param_file.close();

		//Vec3 pos = Vec3(translationSpeed*(1.0f - cosf(time))+x_trans, glass_top_elevation, 0.0f);
		//Vec3 prevPos = Vec3(translationSpeed*(1.0f - cosf(lastTime))+x_trans, glass_top_elevation, 0.0f);

		Vec3 prevPos = Vec3(x_translation + x_trans, glass_top_elevation + y_translation, 0.0f);

		if (theta < (stop_angle*3.14/180) & time > 0) {
			x_translation = x_translation + (time-lastTime)*translationSpeed;
			y_translation = y_translation + (time-lastTime)*2*translationSpeed;
		}

		Vec3 pos = Vec3(x_translation + x_trans, glass_top_elevation + y_translation, 0.0f);
		

		//Vec3 pos = Vec3(translationSpeed*(1.0f - cosf(time)), glass_height, 0.0f);
		//Vec3 prevPos = Vec3(translationSpeed*(1.0f - cosf(lastTime)), glass_height, 0.0f);
		
		Quat rot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -theta); //1.0f - cosf(theta)
		Quat prevRot = QuatFromAxisAngle(Vec3(0.0f, 0.0f, 1.0f), -real_prev_theta); //1.0f - cosf(theta)

		//AddTriangleMesh(mesh, recieve_pos, recieve_rot, 1.0f);
		AddTriangleMesh(pouring_mesh, pos, rot, 1.0f);

		g_buffers->shapePrevPositions[1] = Vec4(prevPos, 0.0f);
		g_buffers->shapePrevRotations[1] = prevRot;	


		UpdateShapes();
	}
};
