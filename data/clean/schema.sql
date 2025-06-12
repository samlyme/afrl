CREATE TABLE FPVPath (
    id INTEGER PRIMARY KEY ASC,
    name TEXT,
    source TEXT
);

-- # timestamp tx ty tz qx qy qz qw
CREATE TABLE Groundtruth (
    pathId INTEGER,
    timestamp REAL,
    tx REAL,
    ty REAL,
    tz REAL,
    qx REAL,
    qy REAL,
    qz REAL,
    qw REAL,
    FOREIGN KEY (pathId) REFERENCES FPVPath(id)
);

-- # timestamp ang_vel_x ang_vel_y ang_vel_z lin_acc_x lin_acc_y lin_acc_z
CREATE TABLE IMU (
    pathId INTEGER,
    timestamp REAL,
    ang_vel_x REAL,
    ang_vel_y REAL,
    ang_vel_z REAL,
    lin_acc_x REAL,
    lin_acc_y REAL,
    lin_acc_z REAL,
    FOREIGN KEY (pathId) REFERENCES FPVPath(id)
);

-- # id timestamp image_name
CREATE TABLE LeftImage (
    pathId INTEGER,
    timestamp REAL,
    imageName TEXT,
    FOREIGN KEY (pathId) REFERENCES FPVPath(id)
);

-- # id timestamp image_name
CREATE TABLE RightImage (
    pathId INTEGER,
    timestamp REAL,
    imageName TEXT,
    FOREIGN KEY (pathId) REFERENCES FPVPath(id)
);