TOKEN_ID="" # User Token in arnord web page

parse_commandline() {
  IFS="="
  while [[ $# -gt 0 ]]; do
    item=($1)
    key=${item[0]}
    value=${item[1]}
    case $key in
    --token_id)
      TOKEN_ID=$value
      shift
      ;;
    --train_id)
      TRAIN_ID=$value
      shift
      ;;
    --infer_id)
      INFER_ID=$value
      shift
      ;;
    --remote_save_root)
      REMOTE_SAVE_ROOT=$value
      shift
      ;;
    --use_lm_jointer)
      USE_LM_JOINTER=$value
      shift
      ;;
    --use_stream)
      USE_STREAM=$value
      shift
      ;;
    --*)
      shift
      ;;
    esac
  done
  unset IFS
}

parse_commandline "$@"
TRAIN_CONTENT=$(curl -v -X GET -H "Authorization: Token ${TOKEN_ID}" https://arnold-api.byted.org/api/v3/trials/${TRAIN_ID}/ 2>/dev/null)
INFER_CONTENT=$(curl -v -X GET -H "Authorization: Token ${TOKEN_ID}" https://arnold-api.byted.org/api/v3/trials/${INFER_ID}/ 2>/dev/null)
python3 scripts/quant_config_auto_gen.py --train_content "${TRAIN_CONTENT}" --infer_content "${INFER_CONTENT}" --remote_save_root "${REMOTE_SAVE_ROOT}" --use_lm_jointer "${USE_LM_JOINTER}" --use_stream "${USE_STREAM}"
