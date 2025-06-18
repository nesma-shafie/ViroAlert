import { Router } from "express";
import { signup, login, logout } from "../controllers/authController.js";
import { validateMiddleware } from "../middlewares/validateMiddleware.js";
import { signupSchema, loginSchema } from "../validation/authSchema.js";
import { auth } from "../middlewares/auth.js";
const authRouter = Router();

authRouter.route("/signup").post(validateMiddleware(signupSchema), signup);

authRouter.route("/login").post(validateMiddleware(loginSchema), login);

authRouter.route('/logout').post(auth, logout);

export default authRouter;
