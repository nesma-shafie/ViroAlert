import pkg from "@prisma/client";
const { PrismaClient } = pkg;

const prisma = new PrismaClient();
export default prisma;

export const checkIfTokenIsActive = async (token) => {
    const tokenRecord = await prisma.webTokens.findUnique({
        where: { token },
    });

    return !!tokenRecord;
};
