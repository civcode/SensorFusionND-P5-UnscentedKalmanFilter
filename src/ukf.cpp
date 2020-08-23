#include "ukf.h"
#include "Eigen/Dense"

using std::cout;
using std::endl;
using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  

  // Process noise standard deviation

  // Longitudinal acceleration in m/s^2
  // Bicicle max acceleration ~ 1.5 m/s^2
  std_a_ = 0.75; //0.75
  
  // Yaw acceleration in rad/s^2
  // Bicicle max angular acceleration ~ 3.14 rad/s^2
  std_yawdd_ = 1.5; //1.5

  // State dimensions
  n_x_ = 5;

  // Augmented state dimensions
  n_aug_ = 7;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

  // Measurement matrix for lidar measurements
  H_lidar_ = MatrixXd(2, n_x_);
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  // Measurement covariance matrix for lidar measurements
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  // Matrix for sigma points in measurement space
  Zsig_ = MatrixXd(3, 2*n_aug_+1);

  // Lidar NIS limit
  nis_lidar_limit_ = 5.991;

  // Radar NIS limit
  nis_radar_limit_ = 7.815;
    
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  const VectorXd rm = meas_package.raw_measurements_;

  // initialization
  if (!is_initialized_) {
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      //cout << "first measuremen: Radar" << endl;
      //cout << "rm.size(): " << rm.size() << endl;
      //cout << "rm: " << rm << endl;

      //compute px and py
      double rho = rm(0);
      double phi = rm(1);

      // initial state mean
      x_ << rho * cos(phi), rho * sin(phi), 0, 0, 0; 

      // initial process covariance matrix
      P_ << .05, 0, 0, 0, 0,
            0, .05, 0, 0, 0,
            0, 0, 0.1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1; 

      is_initialized_ = true;
      time_us_ = meas_package.timestamp_;
    
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      //cout << "first measuremen: Lidar" << endl;
      //cout << "rm.size(): " << rm.size() << endl;
      //cout << "rm: " << rm << endl;

      // initial state mean
      x_ << rm(0), rm(1), 0, 0, 0; 

      // initial process covariance matrix
      P_ << std_laspx_*std_laspx_, 0, 0, 0, 0,
            0, std_laspy_*std_laspy_, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1; 

      is_initialized_ = true;
      time_us_ = meas_package.timestamp_;
    }  
    return;
    
  }

  // Prediction and Update

  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {

    //cout << "*** radar measurement ***" << endl;

    Prediction(dt);

    UpdateRadar(meas_package);

    time_us_ = meas_package.timestamp_;

  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {

    //cout << "*** laser measurement ***" << endl;

    Prediction(dt);

    UpdateLidar(meas_package);

    time_us_ = meas_package.timestamp_;
  }

  // normalize yaw angle
  if (fabs(x_(3)) > M_PI) {
    
    while (x_(3)> M_PI) x_(3)-=2.*M_PI;
    while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

  }

  //cout << "x = " << endl;
  //cout << x_ << endl;  
  
}

void UKF::Prediction(double delta_t) {
  /**
   * Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
   
  // 1. generate sigma points -> Xsig_aug
  // 2. transform sigma points -> Xsig_pred_
  // 3. predict mean and covariance
  
  // Generate augmented sigma points

  // create augmented state mean
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create noise covariance matrix
  MatrixXd Q = MatrixXd(2, 2);

  // create sigma points representing the augmented posterior state distribution (k|k)
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // set augmented state mean vector and covariance matrix
  x_aug.setZero();
  x_aug.head(5) = x_;

  /* P_aug =  [ P_   0 ]
              [ 0    Q ] */

  P_aug.setZero();
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // get matrix A, defined as A.transpose()*A = P_aug
  MatrixXd A = P_aug.llt().matrixL();

  // set augmented sigma points
  Xsig_aug.col(0) = x_aug;

  for (int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  // Transform augmented sigma points

  // transform sigma points by propagating them through the process model
  for (int i=0; i<Xsig_aug.cols(); i++) {

    double p_x      = Xsig_aug(0,i);
    double p_y      = Xsig_aug(1,i);
    double v        = Xsig_aug(2,i);
    double yaw      = Xsig_aug(3,i);
    double yawd     = Xsig_aug(4,i);
    double nu_a     = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p; 

    // prevent division by zero
    if (fabs(yawd) < 0.001) {
      //cout << "*** yaw-rate is zero ***" << endl;
      px_p = p_x + v * cos(yaw) * delta_t;
      py_p = p_y + v * sin(yaw) * delta_t;
    } else {
      px_p = p_x + v/yawd * ( sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (-cos(yaw + yawd * delta_t) + cos(yaw));
    }

    // deterministic part of the prediction
    double v_p    = v;
    double yaw_p  = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p   = px_p   + (delta_t*delta_t) * cos(yaw) * nu_a / 2.;
    py_p   = py_p   + (delta_t*delta_t) * sin(yaw) * nu_a / 2.;
    v_p    = v_p    + delta_t * nu_a;
    yaw_p  = yaw_p  + (delta_t*delta_t) * nu_yawdd / 2.;
    yawd_p = yawd_p + delta_t * nu_yawdd;

    // write sigma points
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  // Predict mean and covariance

  // weight vector
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  double weight_1 = 1/(2*(lambda_+n_aug_));

  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; i++)
    weights_(i) = weight_1;

  // predict state mean
  x_.setZero();  
  for (int i=0; i<2*n_aug_+1; i++) {
    x_ += weights_(i) * Xsig_pred_.col(i);

    //angle normalization
    while (x_(3)> M_PI) x_(3)-=2.*M_PI;
    while (x_(3)<-M_PI) x_(3)+=2.*M_PI;

    /*
    if (fabs(x_(3)) > M_PI) {
      cout << "*** error / Prediction() - (fabs(x_(3)) > M_PI)" << endl;
      exit(0);
    }
    */
  }

  // predict process covariance matrix
  P_.setZero();
  for (int i=0; i<2*n_aug_+1; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
  
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  
  //cout << "*** update laser ***" << endl;

  // get measurement data
  VectorXd z = VectorXd(2);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1);

  // perform standard Kalman filter update
  VectorXd y = z - H_lidar_ * x_;
  MatrixXd S = H_lidar_ * P_ * H_lidar_.transpose() + R_lidar_;
  MatrixXd K = P_ * H_lidar_.transpose() * S.inverse();
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  
  x_ = x_ + K * y;
  P_ = (I - K * H_lidar_) * P_; 

  /*
  if (fabs(x_(3)) > M_PI) {
    cout << "*** error / UpdateLidar() - (fabs(x_(3)) > M_PI)" << endl;
    exit(0);
  }
  */

  // calculate NIS
  double epsilon = y.transpose() * S.inverse() * y;

  nis_vals_lidar_.push_back(epsilon);

  //cout << "*** NIS = " << epsilon << endl;
  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  
  //cout << "*** update radar ***" << endl;

  // transform sigma points to measurement space
  for (int i=0; i<2*n_aug_+1; i++) {

    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    if ((p_x*p_x + p_y*p_y) < 0.000001) {
      cout << "*** error / UpdateRadar() - division by zero ***" << endl;
      exit(0);
    }

    // measurement model
    Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig_(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   
  }

  // mean predicted measurement
  VectorXd z_pred = VectorXd(3);
  z_pred.setZero();
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
      z_pred = z_pred + weights_(i) * Zsig_.col(i);
      /*
      if (fabs(z_pred(1)) > M_PI) {
        cout << "*** error radar update M_PI ***" << endl;
        exit(0);
      }
      */
  }

  // innovation covariance matrix S
  MatrixXd S = MatrixXd(3,3);
  S.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;

    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }  

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(3,3);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;


  // UKF update

  MatrixXd Tc = MatrixXd(n_x_, 3);
  // calculate cross correlation matrix
  Tc.setZero();
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // residual
    VectorXd z_diff = Zsig_.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // get measurement data
  VectorXd z = VectorXd(3);
  z << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), meas_package.raw_measurements_(2);

  while (z(1)> M_PI) z(1)-=2.*M_PI;
  while (z(1)<-M_PI) z(1)+=2.*M_PI;

  /*
  if (fabs(z(1)) > M_PI) {
    cout << "*** error fabs(z(3)) > M_PI ***" << endl;
    exit(0);
  }
  */

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  
  // calculate NIS
  double epsilon = z_diff.transpose() * S.inverse() * z_diff;

  nis_vals_radar_.push_back(epsilon);

  //cout << "*** NIS = " << epsilon << endl; 
}