import prisma from '../prisma.js';

const checkIfTokenIsActive = async (token) => {
    const tokenRecord = await prisma.webTokens.findFirst({
        where: { token },
    });

    return !!tokenRecord;
};

export { checkIfTokenIsActive };
