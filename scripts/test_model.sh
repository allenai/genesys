


# Parse command line arguments
no_check=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--no-check)
            no_check=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Run the check only if no_check is false
if [ "$no_check" = false ]; then
    python -m model_discovery.model.library.tester --mode check "$@"
fi
python -m model_discovery.model.library.tester --mode run "$@"
