import { Router } from "express";
import { predictHost,predictAntivirus } from "../controllers/userController.js";
import { auth } from "../middlewares/auth.js";
import multer from 'multer';
const upload = multer({ dest: 'uploads/' });

const userRouter = Router();

userRouter.route('/predictHost').post(auth,upload.single('file'), predictHost);
userRouter.route('/predictAntiVirus').post(auth, upload.single('file'), predictAntivirus);





export default userRouter;
