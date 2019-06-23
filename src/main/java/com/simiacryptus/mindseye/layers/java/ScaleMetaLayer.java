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

import com.google.gson.JsonObject;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class ScaleMetaLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ScaleMetaLayer.class);

  public ScaleMetaLayer() {
  }

  protected ScaleMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public static ScaleMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ScaleMetaLayer(json);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0];
    final Result in1 = inObj[1];
    final TensorList data0 = in0.getData();
    final TensorList data1 = in1.getData();
    final int itemCnt = data0.length();
    final Tensor data10 = data1.get(0);
    final Tensor[] tensors = IntStream.range(0, itemCnt).mapToObj(dataIndex -> data0.get(dataIndex).mapIndex((v, c) -> v * data10.get(c))).toArray(i -> new Tensor[i]);
    data1.freeRef();
    data0.freeRef();
    Tensor tensor0 = tensors[0];
    tensor0.addRef();
    return new Result(TensorArray.wrap(tensors), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (in0.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(data.stream().map(t -> {
          @Nullable Tensor tensor = t.mapIndex((v, c) -> {
            return v * data10.get(c);
          });
          t.freeRef();
          return tensor;
        }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
      if (in1.isAlive()) {
        @Nullable final Tensor passback = tensor0.mapIndex((v, c) -> {
          return IntStream.range(0, itemCnt).mapToDouble(i -> data.get(i).get(c) * data.get(i).get(c)).sum();
        });
        tensor0.freeRef();
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, data.length())
            .mapToObj(i -> i == 0 ? passback : passback.map(v -> 0)).toArray(i -> new Tensor[i]));
        in1.accumulate(buffer, tensorArray);
      }
      data.freeRef();
    }) {

      @Override
      protected void _free() {
        data10.freeRef();
        Arrays.stream(inObj).forEach(ReferenceCountingBase::freeRef);
      }

      @Override
      public boolean isAlive() {
        return in0.isAlive() || in1.isAlive();
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
