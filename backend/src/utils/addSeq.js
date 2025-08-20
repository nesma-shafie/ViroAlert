import fs from 'fs';
import csv from 'csv-parser';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

async function importCSV(path) {
  const results = [];

  // Read CSV file
  fs.createReadStream(path)
    .pipe(csv())
    .on('data', (data) => {
      // Basic validation (skip empty rows)
      if (data.ID && data.Data) {
        results.push({
          label: data.ID.trim(),
          sequence: data.Data.trim(),
        });
      }
    })
    .on('end', async () => {
      console.log(`Importing ${results.length} sequences...`);

      for (const row of results) { // Limit to first 100 rows for testing
        try {
          await prisma.sequence.create({
            data: row,
          });
        //   console.log(✅ Inserted: ${row.label});
        } catch (err) {
          console.error('❌ Failed to insert ${row.label}:', err.message);
        }
      }

      await prisma.$disconnect();
      console.log('✅ Import completed.');
    });
}

export { importCSV };