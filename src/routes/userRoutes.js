import { Router } from "express";
import { predictHost, predictAntivirus, topAntivirus, predictHost_ML ,generateAntiVirus,align} from "../controllers/userController.js";
import { auth } from "../middlewares/auth.js";
import multer from 'multer';
const upload = multer({ dest: 'uploads/' });

/**
 * @swagger
 * /user/predictHost-ML:
 *   post:
 *     tags:
 *       - User
 *     summary: Predict host (human or non-human) using a machine learning model from a virus sequence
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
 *                 description: Upload a FASTA file containing the virus sequence
 *     responses:
 *       200:
 *         description: Host prediction result using ML model
 *         content:
 *           application/json:
 *             example:
 *               status: success
 *               data:
 *                 prediction: [1]
 *                 probability: [0.987654321]
 *                 img: "base64_encoded_image_string"
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
 *                 img: "sdfdgdfgergrgr"
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
 *         description: Top antiviral drug predictions with scores
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   type: object
 *                   properties:
 *                     top_smiles:
 *                       type: array
 *                       items:
 *                         type: array
 *                         items:
 *                           oneOf:
 *                             - type: string
 *                               description: SMILES string
 *                             - type: number
 *                               format: float
 *                               description: Predicted pIC50 score
 *             example:
 *               status: success
 *               data:
 *                 top_smiles:
 *                   - ["CCc1cn([C@@H]2O[C@H](CNC(=O)C3...)c1=O", 11.49]
 *                   - ["O=C(COc1ccc(Cl)cc1C(=O)c1ccccc1)Nc1ccccc1", 11.42]
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

/**
 * @swagger
 * /user/generateAntiVirus:
 *   post:
 *     tags:
 *       - User
 *     summary: Generate antiviral drug candidates for a virus sequence
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
 *                 description: Virus sequence (required if file not provided)
 *     responses:
 *       200:
 *         description: Generated antiviral drug candidates
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: success
 *                 data:
 *                   type: object
 *                   description: Generated antiviral compounds data
 *             example:
 *               status: success
 *               data:
 *                 drugs:
 *                   - ["CCc1cn([C@@H]2O[C@H](CNC(=O)C3...)c1=O", 11.49]
 *                   - ["O=C(COc1ccc(Cl)cc1C(=O)c1ccccc1)Nc1ccccc1", 11.42]
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
 * /user/align:
 *   get:
 *     tags:
 *       - User
 *     summary: Align input sequence with known sequences and find closest matches
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: seq1
 *         schema:
 *           type: string
 *         description: Input sequence for alignment (optional if file is provided)
 *         example: PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRW...
 *     requestBody:
 *       required: false
 *       content:
 *         multipart/form-data:
 *           schema:
 *             type: object
 *             properties:
 *               file:
 *                 type: string
 *                 format: binary
 *                 description: (Optional) Upload FASTA file containing sequence
 *     responses:
 *       200:
 *         description: Sequence alignment results with closest matches
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 input:
 *                   type: string
 *                   description: Input sequence that was aligned
 *                 closest_matches:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       label:
 *                         type: string
 *                         description: Label/identifier of the matched sequence
 *                       sequence:
 *                         type: string
 *                         description: The matched sequence
 *                       jaccard:
 *                         type: number
 *                         format: float
 *                         description: Jaccard similarity score
 *                       score:
 *                         type: number
 *                         format: float
 *                         description: Alignment score
 *             example:
 *               input: PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRW...
 *               closest_matches:
 *                 - label: HIV-1_sequence_001
 *                   sequence: PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRW...
 *                   jaccard: 0.85
 *                   score: 92.5
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
userRouter.route('/predictHost').post(auth, upload.single('file'), predictHost);
userRouter.route('/predictHost-ML').post(auth, upload.single('file'), predictHost_ML);
userRouter.route('/predictAntiVirus').post(auth, upload.single('file'), predictAntivirus);
userRouter.route('/generateAntiVirus').post(auth, upload.single('file'), generateAntiVirus);
userRouter.route('/topAntiVirus').post(auth, upload.single('file'), topAntivirus);
userRouter.route('/align').get(auth,upload.single('file'), align)




export default userRouter;
