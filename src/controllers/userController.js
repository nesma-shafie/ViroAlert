import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';

const predictHost = async (req, res) => {
    const filePath = req.file.path;  // assuming multer stores it here
    const fileName = req.file.originalname;

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), fileName);

    const fastapiResponse = await axios.post('http://localhost:8000/predict-host', form, {
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


    const fastapiResponse = await axios.post('http://localhost:8000/predict-antivirus', form, {
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



    const fastapiResponse = await axios.post('http://localhost:8000/top-antivirus', form, {
      headers: form.getHeaders(),
    });

    res.status(200).json({
      status: 'success',
      data: fastapiResponse.data,
    });
  }


export  {predictHost, predictAntivirus ,topAntivirus};