import jwt from "jsonwebtoken";
import { getUserBasicInfoByUUID, checkToken } from "../services/userService.js";

const auth = async (req, res, next) => {
    let token = null;
   
  if (req.header("Authorization")) {
    token = req.header("Authorization").replace("Bearer ", "");
  } else {
    return res
      .status(401)
      .json({ status: "fail", message: "no token provided" });
  }

  if (!token) {
    return next(new AppError("no token provided", 401));
  }

  
  let decode = null;
  try {
    decode = jwt.verify(token, process.env.JWT_SECRET);
  } catch (error) {
    return res.status(401).json({
      status: "fail",
      message: "token not valid 11",
    });
  }
  // 1) get user id
  const userId = decode.id;
  // 2) check user existance
  const user = await getUserBasicInfoByUUID(userId);

  if (!user)
    return res.status(404).json({
      status: "fail",
      message: "no user found",
    });

  token = checkToken(token);
  if (!token) {
    return res.status(401).json({
      status: "fail",
      message: "token not valid 222",
    });
  }

  req.user = user;
  req.token = token;
  next();
};

export { auth };
