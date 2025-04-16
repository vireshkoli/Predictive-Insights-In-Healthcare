import React from 'react'

const Result = (props) => {
  if (props.percent < 50) {
    return (
      <div className='flex justify-center my-20 w-full h-screen'>
        <h1 className='text-5xl font-extrabold mt-4 mb-9 text-white'>You Don't have {props.disease} by: {100 - props.percent}%</h1>
      </div>
    )
  } else {
    return (
      <div className='flex justify-center my-20 w-full h-screen'>
          <h1 className='text-5xl font-extrabold mt-4 mb-9 text-white'>The Percentage that you have {props.disease} is: {props.percent}%</h1>
      </div>
    )
  }
}

export default Result