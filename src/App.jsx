import { useState, useEffect } from 'react';

import './App.css'

import {Routes, Route} from 'react-router-dom';

import HomePage from './Components/HomePage/HomePage';
import Login from './Components/Login/Login';
import Register from './Components/Login/Register';
import Disease from "./Components/Disease/Disease";

import axios from 'axios';

function App() {

  // useEffect(() => {
  //   async function getData() {
  //     try {
  //       const response = await axios.get('http://localhost:5000/api/data');
  //       console.log(response);
  //     } catch (error) {
  //       console.error('Error fetching posts:', error);
  //     }
  //   }

  //   async function postData() {
  //     const x = 1;
  //     const y = 'hello';
  //     try {
  //       const response = await axios.post('http://127.0.0.1:5000/api/data', { 'var1': x, 'var2': y });
  //       // alert(`POST Request Successful!\nResponse: ${JSON.stringify(response.data)}`);
  //       console.log(response);
  //     } catch (error) {
  //       console.error('Error making POST request:', error);
  //     }
  //   }

  //   getData();
  //   postData();
  // }, [])

  return (
    <Routes>
      <Route path='/' element={<HomePage />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/disease" element={<Disease/>} />
    </Routes>
    // <h1>Hello World</h1>
  )
}

export default App
