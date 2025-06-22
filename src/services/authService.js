import prisma from '../prisma.js';

const checkIfTokenIsActive = async (token) => {
    const tokenRecord = await prisma.webTokens.findFirst({
        where: { token },
    });

    return !!tokenRecord;
};

const removeToken = async (token) => {
  const deletedToken = await prisma.webTokens.deleteMany({
    where: { token: token },
  });
  return deletedToken;
};


export { checkIfTokenIsActive ,removeToken};
