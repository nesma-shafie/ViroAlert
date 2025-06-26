'use client';
import { Drug } from '@/types';
import { createContext, useContext, useState, ReactNode } from 'react';

type FormContextType = {
    drugs: Drug[];
    setdrugs: (drugs: Drug[]) => void;
};

const FormContext = createContext<FormContextType | undefined>(undefined);

export const useFormContext = () => {
    const context = useContext(FormContext);
    if (!context) throw new Error('useFormContext must be used inside FormProvider');
    return context;
};

export const FormProvider = ({ children }: { children: ReactNode }) => {
    const [drugs, setdrugs] = useState<Drug[]>([]);
    return (
        <FormContext.Provider value={{ drugs, setdrugs }}>
            {children}
        </FormContext.Provider>
    );
};
