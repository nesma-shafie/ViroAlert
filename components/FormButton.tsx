import { Button } from "@/components/ui/button";

export default function FormButton({ isLoading, text }: { isLoading: boolean, text: string }) {
    return (
        <Button
            type="submit"
            disabled={isLoading}
            className="w-56 virogen-blue hover:virogen-light-blue text-white"
        >
            {text}
        </Button>
    )
}