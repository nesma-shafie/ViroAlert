import { Router } from "express";
import { predictHost,predictAntivirus,topAntivirus ,generateAntiVirus,align} from "../controllers/userController.js";
import { auth } from "../middlewares/auth.js";
import multer from 'multer';
const upload = multer({ dest: 'uploads/' });

/**
 * @swagger
 * /user/predictHost:
 *   post:
 *     tags:
 *       - User
 *     summary: Predict host (human or non-human) from a virus sequence
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             required:
 *               - file
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: Upload a FASTA file
 *     responses:
 *       200:
 *         description: Host prediction result
 *         content:
 *           application/json:
 *             example:
 *               status: success
 *               data:
 *                 prediction: [1]
 *                 probability: [0.9999639987945557]
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: Token not valid
 *       404:
 *         description: No user found
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: No user found
 */


/**
 * @swagger
 * /user/predictAntiVirus:
 *   post:
 *     tags:
 *       - User
 *     summary: Predict effectiveness of a drug against a virus (pIC50)
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: (Optional) Upload FASTA file
 *               virus:
 *                 type: string
 *                 example: PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRW...
 *               smiles:
 *                 type: string
 *                 example: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O
 *     responses:
 *       200:
 *         description: Antivirus prediction result
 *         content:
 *           application/json:
 *             example:
 *               status: success
 *               data:
 *                 pIC50: 4.989653587341309
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: Token is no longer active
 *       404:
 *         description: No user found
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: No user found
 */
/**
 * @swagger
 * /user/topAntiVirus:
 *   post:
 *     tags:
 *       - User
 *     summary: Get top predicted antiviral drug candidates for a virus
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: (Optional) Upload FASTA file
 *               virus:
 *                 type: string
 *                 example: PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRW...
 *     responses:
 *       200:
 *         description: Top antiviral drug predictions
 *         content:
 *           application/json:
 *             example:
 *               status: success
 *               data:
 *                 top_smiles:
 *                   - "CCc1cn([C@@H]2O[C@H](CNC(=O)C3...)c1=O"
 *                   - "O=C(COc1ccc(Cl)cc1C(=O)c1ccccc1)Nc1ccccc1"
 *       401:
 *         description: Unauthorized
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: No token provided
 *       404:
 *         description: No user found
 *         content:
 *           application/json:
 *             example:
 *               status: fail
 *               message: No user found
 */

const userRouter = Router();
userRouter.route('/predictHost').post(auth,upload.single('file'), predictHost);
userRouter.route('/predictAntiVirus').post(auth, upload.single('file'), predictAntivirus);
userRouter.route('/generateAntiVirus').post(auth, upload.single('file'), generateAntiVirus);
userRouter.route('/topAntiVirus').post(auth, upload.single('file'), topAntivirus);
userRouter.route('/align').get(auth,upload.single('file'), align)




export default userRouter;
