/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Initialize all particles to first position (based on estimates of 
  // x, y, theta and their uncertainties from GPS) and all weights to 1. 
  num_particles = 50;
  default_random_engine gen;
  Particle particle;
  
  // normal (Gaussian) distribution based on GPS estimates.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; i++) {
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);

    // Normalize theta
    particles[i].theta = normalizeTheta(particles[i].theta);
    cout << "Particle " << i + 1 << " " << particles[i].x << " " << particles[i].y << " " << particles[i].theta << endl;
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Adding measurements and noise to each particle
  default_random_engine gen;
  double vdt;
  double v_yawrate;
  double yawrate_dt;

  vdt = velocity * delta_t;	
  v_yawrate = velocity/yaw_rate;
  yawrate_dt = yaw_rate * delta_t;
  
  for (int i = 0; i < num_particles; i++) {
    if(yaw_rate == 0) {
      particles[i].x += vdt * cos(particles[i].theta);
      particles[i].y += vdt * sin(particles[i].theta);
    } else {
      particles[i].x += v_yawrate * (sin(particles[i].theta + yawrate_dt) - sin(particles[i].theta));
      particles[i].y += v_yawrate * (cos(particles[i].theta) - cos(particles[i].theta + yawrate_dt));
      particles[i].theta += yawrate_dt;
    }

    normal_distribution<double> measnoise_x(particles[i].x, std_pos[0]);
    normal_distribution<double> measnoise_y(particles[i].y, std_pos[1]);
    normal_distribution<double> measnoise_theta(particles[i].theta, std_pos[2]);
    double noise_x = measnoise_x(gen); 
    double noise_y = measnoise_y(gen); 
    double noise_theta = measnoise_theta(gen);

    particles[i].x += noise_x;
    particles[i].y += noise_y;
    particles[i].theta += noise_theta;

    // Normalize theta
    particles[i].theta = normalizeTheta(particles[i].theta);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  int num_observations = observations.size();
  int num_predictions = predicted.size();
  double match;
  double currMatch;
  int matchIndex;

  // Find closest matching observation to each landmark
  // using nearnest neighbour technique 
  for (int i = 0; i < num_observations; i++) {
    match = 0;
    matchIndex = 0;
    for (int j = 0; j < num_predictions; j++) {
      // Find the Euclidean distance between the predicted measurement and each observation measurement
      currMatch = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
      // Find the closest match
      if(j == 0) {
        match = currMatch;
        matchIndex = j;
      } else {
        if(currMatch <= match) {
          match = currMatch;
          matchIndex = j;
        }
      }
    }
    observations[i].id = predicted[matchIndex].id; 
    //cout << "Matching landmark for observation " << i << " is " << observations[i].id << endl;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  std::vector<LandmarkObs> realworld_landmark;
  std::vector<LandmarkObs> map_observations;
  LandmarkObs landmark;
  double gauss_norm, mu_x, mu_y, x_obs, y_obs, var_x, var_y, exponent;
  double par_to_land;

  // Convert map_landmarks to LandmarkObs structure 
  realworld_landmark.clear();
  for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
    landmark.x = map_landmarks.landmark_list[i].x_f;
    landmark.y = map_landmarks.landmark_list[i].y_f;
    landmark.id = map_landmarks.landmark_list[i].id_i;
    realworld_landmark.push_back(landmark);
  }
  
  gauss_norm = 1/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
  var_x = std_landmark[0] * std_landmark[0];
  var_y = std_landmark[1] * std_landmark[1];
  
  for (int i=0; i< num_particles; i++) {
    // Initializations required for each update
    map_observations.clear();
    particles[i].weight = 1;

    // Homogenouse transformation of observations from vehicles coordinate system
    // to MAP's coordinate system
    for (int j=0; j< observations.size(); j++) {
      landmark.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
      landmark.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
      // Init each observation id to 0
      landmark.id = 0;
      //cout << "Transformed observation x = " << landmark.x << " y = " << landmark.y << " id = " << landmark.id << endl;
      map_observations.push_back(landmark);
    }
    
    // Associate each observerd landmark with a map landmark
    dataAssociation(realworld_landmark, map_observations);

    // Update the weight of each particle using a mult-variate Gaussian distribution. 
    for (int j=0; j< map_observations.size(); j++) {
      for (int k=0; k< realworld_landmark.size(); k++) { 
        // Check if distance between particle and landmark is within sensor range
        par_to_land = dist(particles[i].x, particles[i].y, realworld_landmark[k].x, realworld_landmark[k].y);

        if((par_to_land < sensor_range) && (map_observations[j].id == realworld_landmark[k].id)) {
          mu_x = realworld_landmark[k].x;
          mu_y = realworld_landmark[k].y;
          x_obs = map_observations[j].x;
          y_obs = map_observations[j].y;
          exponent = ((x_obs-mu_x)*(x_obs-mu_x))/(2*var_x) + ((y_obs-mu_y)*(y_obs-mu_y))/(2*var_y);
          //cout << "landmark[ " << j << "] gauss_norm = " << gauss_norm << " mu_x = " << mu_x << " x_obs = " 
          //     << x_obs << " mu_y = " << mu_y << " y_obs = " << y_obs << " exponent = " << exponent << endl;
          particles[i].weight *= gauss_norm * exp(-exponent);
          break;
        }
      }
    }
    //cout << "particles[" << i << "].weight = " << particles[i].weight << endl;
  }
}

// Resample particles with replacement with probability proportional to their weight. 
void ParticleFilter::resample() {
  int index = 0;
  double beta = 0;
  double max_weight =  0;
  Particle particle;
  std::vector<Particle> resampled;
 
  index = int(((double) rand() / (RAND_MAX)) * num_particles);
  for(int i = 0; i < num_particles; i++) {
    if(particles[i].weight > max_weight)
      max_weight = particles[i].weight;
  }
  
  resampled.clear();
  for(int i = 0; i < num_particles; i++) {
    beta += ((double)rand()/RAND_MAX) * 2.0 * max_weight;
    while (beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    particle.x = particles[index].x;
    particle.y = particles[index].y;
    particle.theta = particles[index].theta;
    particle.weight = particles[index].weight;
    //cout << "Resampled index = " << index << " max weight = " << max_weight << endl; 
    resampled.push_back(particle);
    //resampled.push_back(particles[index]);
  }
  for(int i = 0; i < num_particles; i++) {
    particles[i].x = resampled[i].x;
    particles[i].y = resampled[i].y;
    particles[i].theta = resampled[i].theta;
    particles[i].weight = resampled[i].weight;
  }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
