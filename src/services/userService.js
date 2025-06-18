import jwt from "jsonwebtoken";
const JWT_SECRET = process.env.JWT_SECRET; // Use .env for production
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN; // Or '1h', '30m', etc.

import pkg from "@prisma/client";
const { PrismaClient } = pkg;

const prisma = new PrismaClient();
export default prisma;

/**
 * Get count of users that match either the email or username
 */
const getUsersCountByEmailUsername = async (email, username) => {
  const count = await prisma.user.count({
    where: {
      OR: [
        { email: email.toLowerCase() },
        { username: username.toLowerCase() },
      ],
    },
  });
  return count;
};

/**
 * Create a new user
 */
const createNewUser = async (email, username, name, hashedPassword) => {
  return await prisma.user.create({
    data: {
      email: email.toLowerCase(),
      username: username.toLowerCase(),
      name,
      password: hashedPassword,
    },
  });
};

/**
 * Get basic user info by ID (for token & profile return)
 */
const getUserBasicInfoByUUID = async (id) => {
  return await prisma.user.findUnique({
    where: { id },
    select: {
      id: true,
      email: true,
      username: true,
      name: true,
    },
  });
};

/**
 * Get basic user info by username
 */
const getUserBasicInfoByUsername = async (username) => {
  return await prisma.user.findUnique({
    where: {
      username: username.toLowerCase(),
    },
    select: {
      id: true,
      username: true,
      name: true,
      email: true,
    },
  });
};

/**
 * Get hashed password by user ID
 */
const getUserPassword = async (userId) => {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { password: true },
  });
  return user?.password || null;
};

const generateToken = async (userId) => {
  const token = await jwt.sign({ id: userId }, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN,
  });
  await prisma.WebTokens.create({
    data: {
      userID: userId,
      token: token,
    },
  });
  return token;
};

const checkToken = async (token) => {
  const userToken = await prisma.WebTokens.findFirst({
    where: { token },
  });
  return userToken;
};

const removeToken = async (token) => {
  const deletedToken = await prisma.webTokens.deleteMany({
    where: { token: token },
  });
  return deletedToken;
};

export {
  getUsersCountByEmailUsername,
  createNewUser,
  getUserBasicInfoByUUID,
  getUserBasicInfoByUsername,
  getUserPassword,
  generateToken,
  removeToken,
  checkToken,
};
