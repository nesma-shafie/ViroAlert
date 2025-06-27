import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import { getKmers, jaccardSimilarity, align2seq } from '../utils/align.js';
import prisma from '../prisma.js'; // Assuming you have a Prisma client setup
const predictHost = async (req, res) => {
  const filePath = req.file.path;  // assuming multer stores it here
  const fileName = req.file.originalname;

  const form = new FormData();
  form.append('file', fs.createReadStream(filePath), fileName);

  const url = process.env.FASTAPI_URL + '/predict-host'; // Ensure you have this environment variable set
  const fastapiResponse = await axios.post(url, form, {
    headers: form.getHeaders(),
  });

  res.status(200).json({
    status: 'success',
    data: fastapiResponse.data,
  });
}
const predictHost_ML = async (req, res) => {
  const filePath = req.file.path;  // assuming multer stores it here
  const fileName = req.file.originalname;

  const form = new FormData();
  form.append('file', fs.createReadStream(filePath), fileName);

  const url = process.env.FASTAPI_URL + '/predict-host-ML'; // Ensure you have this environment variable set
  const fastapiResponse = await axios.post(url, form, {
    headers: form.getHeaders(),
  });

  res.status(200).json({
    status: 'success',
    data: fastapiResponse.data,
  });
}

const predictAntivirus = async (req, res) => {
  const form = new FormData();

  // If FASTA file is uploaded
  if (req.file) {
    const filePath = req.file.path;
    const fileName = req.file.originalname;

    form.append('file', fs.createReadStream(filePath), fileName);
  }

  // If virus sequence is provided manually (and not file)
  if (!req.file && req.body.virus) {
    form.append('virus', req.body.virus);
  }

  form.append('smiles', req.body.smiles);


  const url = process.env.FASTAPI_URL + '/predict-antivirus';

  const fastapiResponse = await axios.post(url, form, {
    headers: form.getHeaders(),
  });

  res.status(200).json({
    status: 'success',
    data: fastapiResponse.data,
  });
}

const topAntivirus = async (req, res) => {
  const form = new FormData();

  // If FASTA file is uploaded
  if (req.file) {
    const filePath = req.file.path;
    const fileName = req.file.originalname;

    form.append('file', fs.createReadStream(filePath), fileName);
  }

  // If virus sequence is provided manually (and not file)
  if (!req.file && req.body.virus) {
    form.append('virus', req.body.virus);
  }

  const url = process.env.FASTAPI_URL + '/top-antivirus'; // Ensure you have this environment variable set
  const fastapiResponse = await axios.post(url, form, {
    headers: form.getHeaders(),
  });

  res.status(200).json({
    status: 'success',
    data: fastapiResponse.data,
  });
}

const generateAntiVirus = async (req, res) => {
  const form = new FormData();

  // If FASTA file is uploaded
  if (req.file) {
    const filePath = req.file.path;
    const fileName = req.file.originalname;

    form.append('file', fs.createReadStream(filePath), fileName);
  }

  // If virus sequence is provided manually (and not file)
  if (!req.file && req.body.virus) {
    form.append('virus', req.body.virus);
  }

  const url = process.env.FASTAPI_URL + '/generate-antivirus'; // Ensure you have this environment variable set
  const fastapiResponse = await axios.post(url, form, {
    headers: form.getHeaders(),
  });

  res.status(200).json({
    status: 'success',
    data: fastapiResponse.data,
  });
}
const align = async (req, res) => {
  const form = new FormData();

  // If FASTA file is uploaded
  let input_sequence = '';
  if (req.file) {
    const filePath = req.file.path;
    const fileName = req.file.originalname;

    // form.append('file', fs.createReadStream(filePath), fileName);
    let fileContent = fs.readFileSync(filePath, 'utf8').trim();
    // Check for multiple FASTA headers
    const headerCount = (fileContent.match(/^>/gm) || []).length;
    if (headerCount > 1) {
      return res.status(400).json({
        status: 'error',
        message: 'FASTA file contains more than one sequence (multiple headers found). Please upload a file with a single sequence.'
      });
    }
    if (fileContent.startsWith('>')) {
      // Remove the first line (FASTA header)
      input_sequence = fileContent.split('\n').slice(1).join('').trim();
    } else {
      input_sequence = fileContent;
    }

  }

  // If sequences are provided manually (and not file)
  if (!req.file && req.body.virus) {
    input_sequence = req.body.virus;
  }
  //get the sequence from the DB
  const inputKmers = getKmers(input_sequence, 5); // Can tune k

  // Fetch known sequences
  const knownSequences = await prisma.sequence.findMany();

  // Step 1: Filter by Jaccard similarity
  const filtered = knownSequences
    .map(entry => {
      const targetKmers = getKmers(entry.sequence, 5);
      const jaccard = jaccardSimilarity(inputKmers, targetKmers);
      return { ...entry, jaccard };
    });
    let sim=0.2;
     let    finalfiltered= filtered.filter(entry => entry.jaccard >= 0.2);

    while(filtered.length<5)
   { 
    sim -= 0.05;
    finalfiltered= filtered.filter(entry => entry.jaccard >= sim);
  }
  console.log(filtered.length);
  // Sort by Jaccard descending
  filtered.sort((a, b) => b.jaccard - a.jaccard);

  // Step 2: Run alignment only on the top N from filtered
  const results = filtered.slice(0, 30).map(entry => {
    const {alignedSeq1,alignedSeq2,matchLine,score} = align2seq(input_sequence, entry.sequence); // fast version
    return { label: entry.label, sequence: entry.sequence, jaccard: entry.jaccard, score, alignedSeq1, alignedSeq2, matchLine };
  });

  // Sort by highest score (closest match first)
  results.sort((a, b) => b.score - a.score);

  res.json({
    input: input_sequence,
    closest_matches: results.slice(0, 5) // top 5 matches
  });
}

export { predictHost, predictAntivirus, topAntivirus, predictHost_ML, generateAntiVirus, align };
