/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class TensorConcatLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(TensorConcatLayer.class);
  private int maxBands;

  public TensorConcatLayer() {
    setMaxBands(0);
  }

  protected TensorConcatLayer(@Nonnull final JsonObject json) {
    super(json);
    JsonElement maxBands = json.get("maxBands");
    if (null != maxBands)
      setMaxBands(maxBands.getAsInt());
  }

  public int getMaxBands() {
    return maxBands;
  }

  @Nonnull
  public void setMaxBands(int maxBands) {
    this.maxBands = maxBands;
  }

  @Nonnull
  @SuppressWarnings("unused")
  public static TensorConcatLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new TensorConcatLayer(json);
  }

  @Nullable
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    TensorList temp_09_0009 = inObj[0].getData();
    final int numBatches = temp_09_0009.length();
    temp_09_0009.freeRef();
    assert RefArrays.stream(RefUtil.addRefs(inObj)).allMatch(x -> {
      TensorList temp_09_0010 = x.getData();
      boolean temp_09_0004 = temp_09_0010.length() == numBatches;
      temp_09_0010.freeRef();
      x.freeRef();
      return temp_09_0004;
    }) : "All inputs must use same batch size";
    int[] outputDims = new int[]{RefArrays.stream(RefUtil.addRefs(inObj)).mapToInt(x -> {
      TensorList temp_09_0011 = x.getData();
      int temp_09_0005 = Tensor.length(temp_09_0011.getDimensions());
      temp_09_0011.freeRef();
      x.freeRef();
      return temp_09_0005;
    }).sum()};

    @Nonnull final RefList<Tensor> outputTensors = new RefArrayList<>();
    for (int b = 0; b < numBatches; b++) {
      @Nonnull final Tensor outputTensor = new Tensor(outputDims);
      int pos = 0;
      @Nullable final double[] outputTensorData = outputTensor.getData();
      for (int i = 0; i < inObj.length; i++) {
        TensorList temp_09_0012 = inObj[i].getData();
        @Nullable
        Tensor tensor = temp_09_0012.get(b);
        temp_09_0012.freeRef();
        @Nullable final double[] data = tensor.getData();
        tensor.freeRef();
        RefSystem.arraycopy(data, 0, outputTensorData, pos,
            Math.min(data.length, outputTensorData.length - pos));
        pos += data.length;
      }
      outputTensors.add(outputTensor);
    }
    try {
      return new Result(new TensorArray(outputTensors.toArray(new Tensor[]{})), new Result.Accumulator() {
        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public void accept(@Nullable DeltaSet<UUID> buffer, @Nonnull TensorList data) {
          assert numBatches == data.length();

          @Nonnull final RefList<Tensor[]> splitBatches = new RefArrayList<>();
          for (int b = 0; b < numBatches; b++) {
            @Nullable final Tensor tensor = data.get(b);
            @Nonnull final Tensor[] outputTensors2 = new Tensor[inObj.length];
            int pos = 0;
            for (int i = 0; i < inObj.length; i++) {
              TensorList temp_09_0013 = inObj[i].getData();
              @Nonnull final Tensor dest = new Tensor(temp_09_0013.getDimensions());
              temp_09_0013.freeRef();
              @Nullable
              double[] tensorData = tensor.getData();
              RefSystem.arraycopy(tensorData, pos, dest.getData(), 0,
                  Math.min(dest.length(), tensorData.length - pos));
              pos += dest.length();
              RefUtil.set(outputTensors2, i, dest.addRef());
              dest.freeRef();
            }
            tensor.freeRef();
            splitBatches.add(RefUtil.addRefs(outputTensors2));
            RefUtil.freeRef(outputTensors2);
          }

          data.freeRef();
          @Nonnull final Tensor[][] splitData = new Tensor[inObj.length][];
          for (int i = 0; i < splitData.length; i++) {
            RefUtil.set(splitData, i, new Tensor[numBatches]);
          }
          for (int i = 0; i < inObj.length; i++) {
            for (int b = 0; b < numBatches; b++) {
              Tensor[] tensors = splitBatches.get(b);
              assert tensors != null;
              RefUtil.set(splitData[i], b, tensors[i].addRef());
              RefUtil.freeRef(tensors);
            }
          }

          splitBatches.freeRef();
          for (int i = 0; i < inObj.length; i++) {
            TensorArray wrap = new TensorArray(RefUtil.addRefs(splitData[i]));
            inObj[i].accumulate(buffer == null ? null : buffer.addRef(), wrap.addRef());
            if (0 < wrap.currentRefCount()) {
              RefUtil.freeRef(splitData);
              RuntimeException temp_09_0007 = new RuntimeException(
                  inObj[i].getClass() + " leak: " + wrap.currentRefCount());
              wrap.freeRef();
              if (null != buffer)
                buffer.freeRef();
              throw temp_09_0007;
            }
            wrap.freeRef();
          }
          if (null != buffer)
            buffer.freeRef();
          RefUtil.freeRef(splitData);
        }

        public @SuppressWarnings("unused")
        void _free() {
          super._free();
          RefUtil.freeRef(inObj);
        }
      }) {

        {
          RefUtil.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          for (@Nonnull final Result element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }

        public void _free() {
          RefUtil.freeRef(inObj);
          super._free();
        }
      };
    } finally {
      RefUtil.freeRef(inObj);
      outputTensors.freeRef();
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull
    JsonObject json = super.getJsonStub();
    json.addProperty("maxBands", maxBands);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() { super._free(); }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  TensorConcatLayer addRef() {
    return (TensorConcatLayer) super.addRef();
  }
}
