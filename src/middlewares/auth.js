import jwt from 'jsonwebtoken';
import userService from '../services/userService.js';
import { checkIfTokenIsActive } from '../services/authService.js';

const auth = async (req, res, next) => {
    let token = null;

    // Extract token from Authorization header
    if (req.header('Authorization')) {
        token = req.header('Authorization').replace('Bearer ', '');
    }

    if (!token) {
        return res.status(401).json({
            status: 'fail',
            message: 'No token provided',
        });
    }

    let decode;
    try {
        decode = jwt.verify(token, process.env.JWT_SECRET);
    } catch (error) {
        return res.status(401).json({
            status: 'fail',
            message: 'Token not valid',
        });
    }

    const userId = JSON.parse(decode.id);
    const user = await userService.getUserAllDetailsById(userId);

    if (!user) {
        return res.status(404).json({
            status: 'fail',
            message: 'No user found',
        });
    }

    const isActive = await checkIfTokenIsActive(token);
    if (!isActive) {
        return res.status(401).json({
            status: 'fail',
            message: 'Token is no longer active',
        });
    }

    req.user = user;
    next();
};

export  {auth};
