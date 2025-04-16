import React, { useState } from 'react';
import Diabeties from "./Diabeties";
import LungCancer from './LungCancer';

const Disease = () => {

    const [disease, setDisease] = useState(<></>)

    const idata = ["Tuberculosis(TB)", "Malaria", "Covid19"];
    const cdata = ["Diabetes", "Heart Disease", "Kindey Disease"];
    const cadata = ["Breast Cancer", "Lung Cancer", "Skin Cancer"];

    const candata = [
        {
            data: idata,
            color: "bg-yellow-400"
        }, 
        {
            data: cdata,
            color: "bg-sky-500"
        }, 
        {
            data: cadata,
            color: "bg-red-500"
        }
    ];

    const handleChange1 = (e) => {
        let d = e.target.value;

        if (d === "Tuberculosis(TB)")
            console.log(d);
        else if (d === "Malaria")
            console.log(d);
        else if (d === "Covid19")
            console.log(d)
    }

    const handleChange2 = (e) => {
        let d = e.target.value;

        if (d === "Diabetes")
            setDisease(<Diabeties />);
        else if (d === "Heart Disease")
            console.log(d);
        else if (d === "Kindey Disease")
            console.log(d)
    }

    const handleChange3 = (e) => {
        let d = e.target.value;

        if (d === "Breast Cancer")
            console.log(d);
        else if (d === "Lung Cancer")
            setDisease(<LungCancer />)
        else if (d === "Skin Cancer")
            console.log(d)
    }

  return (
    <div className='flex flex-col bg-gradient-to-b from-[#000] to-[#4b5c60] h-full'>
        <div className='w-full p-4 text-center'>
            <h1 className='text-5xl font-extrabold mt-4 mb-9 text-white'>Select the type of disease you want to check</h1>
            <div className='flex gap-10 justify-around'>
                <select className='select-style bg-yellow-400' onChange={handleChange1}>
                    {idata.map((ele, ind) => {
                        if(ind === 0)
                            return (
                                <>
                                    <option selected>Select the type of infectious disease</option>
                                    <option>{ele}</option>
                                </>
                            )
                        return <option>{ele}</option>
                    })}
                </select>
                <select className='select-style bg-blue-400' onChange={handleChange2}>
                    {cdata.map((ele, ind) => {
                        if(ind === 0)
                            return (
                                <>
                                    <option selected>Select the type of Chronic disease</option>
                                    <option>{ele}</option>
                                </>
                            )
                        return <option>{ele}</option>
                    })}
                </select>
                <select className='select-style bg-red-400' onChange={handleChange3}>
                    {cadata.map((ele, ind) => {
                        if(ind === 0)
                            return (
                                <>
                                    <option selected>Select the type of Cancer</option>
                                    <option>{ele}</option>
                                </>
                            )
                        return <option>{ele}</option>
                    })}
                </select>
            </div>
        </div>
        <div className='w-full mt-5 p-4 text-center'>
            <h1 className='font-extrabold text-5xl inline-block text-white'>Enter the disease symptoms</h1>
            {disease}
        </div>
    </div>
  )
}

export default Disease