<template>
  <div class="mock-banner">
    <v-icon class="mr-2" size="18">mdi-flask-outline</v-icon>
    <span>
      <strong>MOCK BACKEND — NOT PinPoint3D inference.</strong>
      The mask in this session comes from a deterministic geometric rule so the UI can be
      verified end to end. Do not use these results as model output.
    </span>
  </div>
</template>

<script lang="ts" setup>
// Rendered only when VITE_MOCK_BACKEND=true. The banner keeps mock runs unambiguous
// in screenshots and recordings.
</script>

<style scoped>
.mock-banner {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 6px 16px;
  background-color: #5c2e00;
  color: #fff3e0;
  border-bottom: 2px solid #ff8f00;
  font-size: 13px;
  line-height: 1.4;
}
</style>
