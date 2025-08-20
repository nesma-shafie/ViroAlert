import bcrypt from "bcryptjs";
import {
  generateToken,
  getUserBasicInfoByUUID,
  createNewUser,
  getUsersCountByEmailUsername,
  getUserBasicInfoByUsername,
  getUserPassword,
} from "../services/userService.js";

import {removeToken} from "../services/authService.js";

const signup = async (req, res, next) => {
  const { email, password, username } = req.body;
  // 1) check if the username, email is valid
  const usersCount = await getUsersCountByEmailUsername(email, username);

  if (usersCount) {
    return res.status(400).json({
      status: "fail",
      message:
        "There is a user in the database with the same email or username",
    });
  }

  const hashedPassword = await bcrypt.hash(password, 8);
  let user = await createNewUser(email, username, hashedPassword);
  if (!user) {
    return res.status(400).json({
      status: "fail",
      message: "user was not created",
    });
  }

  user = await getUserBasicInfoByUUID(user.id);

  const token = await generateToken(user.id);
  return res.status(200).send({ data: { user, token }, status: "success" });
};

const login = async (req, res, next) => {
  const { password, username } = req.body;

  const user = await getUserBasicInfoByUsername(username);

  if (!user) {
    return res.status(404).json({
      status: "fail",
      message: "no user found ",
    });
  }
  const hashedPassword = await getUserPassword(user.id);
  if (!(await bcrypt.compare(password, hashedPassword))) {
    return res.status(401).json({
      status: "fail",
      message: "wrong password",
    });
  }
  const token =await  generateToken(user.id);
  return res.status(200).send({ data: { user, token }, status: "success" });
};

const logout = async (req, res, next) => {
  const token = req.header("Authorization").replace("Bearer ", "");
  await removeToken(req.token);
  return res.status(200).send({ status: "success" });
};

export { signup, login, logout };