import numpy as np
import math


# Function To Euler Angles To Convert Rotation Vector
def EulerToVector(roll, pitch, yaw):
    alpha, beta, gamma = yaw, pitch, roll
    ca, cb, cg, sa, sb, sg = (
        math.cos(alpha),
        math.cos(beta),
        math.cos(gamma),
        math.sin(alpha),
        math.sin(beta),
        math.sin(gamma),
    )
    r11, r12, r13 = ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg
    r21, r22, r23 = sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg
    r31, r32, r33 = -sb, cb * sg, cb * cg
    theta = math.acos((r11 + r22 + r33 - 1) / 2)
    sth = math.sin(theta)
    kx, ky, kz = (
        (r32 - r23) / (2 * sth),
        (r13 - r31) / (2 * sth),
        (r21 - r12) / (2 * sth),
    )
    return [(theta * kx), (theta * ky), (theta * kz)]


# Function To Convert Rotation Vector To Euler Angles
def VectorToEuler(rx, ry, rz):
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    cth, sth, vth = math.cos(theta), math.sin(theta), 1 - math.cos(theta)
    r11, r12, r13 = (
        kx * kx * vth + cth,
        kx * ky * vth - kz * sth,
        kx * kz * vth + ky * sth,
    )
    r21, r22, r23 = (
        kx * ky * vth + kz * sth,
        ky * ky * vth + cth,
        ky * kz * vth - kx * sth,
    )
    r31, r32, r33 = (
        kx * kz * vth - ky * sth,
        ky * kz * vth + kx * sth,
        kz * kz * vth + cth,
    )
    beta = math.atan2(-r31, math.sqrt(r11 * r11 + r21 * r21))
    if beta > math.radians(89.99):
        beta = math.radians(89.99)
        alpha = 0
        gamma = math.atan2(r12, r22)
    elif beta < -math.radians(89.99):
        beta = -math.radians(89.99)
        alpha = 0
        gamma = -math.atan2(r12, r22)
    else:
        cb = math.cos(beta)
        alpha = math.atan2(r21 / cb, r11 / cb)
        gamma = math.atan2(r32 / cb, r33 / cb)
    return [gamma, beta, alpha]


# Function To Get Rotation Matrix
def GetRotationMatrix(Pose):
    X, Y, Z, Rx, Ry, Rz = Pose[0], Pose[1], Pose[2], Pose[3], Pose[4], Pose[5]
    Rr = VectorToEuler(Rx, Ry, Rz)
    Rx, Ry, Rz = Rr[0], Rr[1], Rr[2]
    M11 = math.cos(Ry) * math.cos(Rz)
    M12 = (math.sin(Rx) * math.sin(Ry) * math.cos(Rz)) - (math.cos(Rx) * math.sin(Rz))
    M13 = (math.cos(Rx) * math.sin(Ry) * math.cos(Rz)) + (math.sin(Rx) * math.sin(Rz))
    M21 = math.cos(Ry) * math.sin(Rz)
    M22 = (math.sin(Rx) * math.sin(Ry) * math.sin(Rz)) + (math.cos(Rx) * math.cos(Rz))
    M23 = (math.cos(Rx) * math.sin(Ry) * math.sin(Rz)) - (math.sin(Rx) * math.cos(Rz))
    M31 = -math.sin(Ry)
    M32 = math.sin(Rx) * math.cos(Ry)
    M33 = math.cos(Rx) * math.cos(Ry)
    return np.stack([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])


# Function To Get Rotation Vector
def GetRotation(RM):
    Ry = math.atan2(-RM[2][0], math.sqrt(RM[0][0] ** 2 + RM[1][0] ** 2))
    Rz = math.atan2(RM[1][0] / math.cos(Ry), RM[0][0] / math.cos(Ry))
    Rx = math.atan2(RM[2][1] / math.cos(Ry), RM[2][2] / math.cos(Ry))
    Rr = EulerToVector(Rx, Ry, Rz)
    Rx, Ry, Rz = Rr[0], Rr[1], Rr[2]
    return [Rx, Ry, Rz]


# Function To Get 3D Point Matrix
def GetPointMatrix(Pose):
    X, Y, Z, Rx, Ry, Rz = Pose[0], Pose[1], Pose[2], Pose[3], Pose[4], Pose[5]
    return np.stack([X, Y, Z])


# Pose_Trans Function
def PoseTrans(Pose1, Pose2):
    P1 = GetPointMatrix(Pose1)
    P2 = GetPointMatrix(Pose2)
    R1 = GetRotationMatrix(Pose1)
    R2 = GetRotationMatrix(Pose2)
    P = np.add(P1, np.matmul(R1, P2))
    R = GetRotation(np.matmul(R1, R2))
    return [P[0], P[1], P[2], R[0], R[1], R[2]]


# Pose_Add Function
def PoseAdd(Pose1, Pose2):
    P1 = GetPointMatrix(Pose1)
    P2 = GetPointMatrix(Pose2)
    R1 = GetRotationMatrix(Pose1)
    R2 = GetRotationMatrix(Pose2)
    P = np.add(P1, P2)
    R = GetRotation(np.matmul(R1, R2))
    return [P[0], P[1], P[2], R[0], R[1], R[2]]

